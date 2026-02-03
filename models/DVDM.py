import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .coattention import co_attention
from .layers import Attention


class DVDMModel(nn.Module):
    """
    Dual-View Debiasing Model for Video Misinformation Detection
    Event-level Counterfactual Reasoning
    Entity-level Causal Intervention
    """

    def __init__(self, bert_model, fea_dim=128, dropout=0.3):
        super(DVDMModel, self).__init__()

        # -------- Text Encoder --------
        self.bert = BertModel.from_pretrained(bert_model)
        self.bert.requires_grad_(False)
        self.text_dim = 768

        # -------- Modality Dimensions --------
        self.img_dim = 4096
        self.video_dim = 4096
        self.audio_dim = 128
        self.entity_dim = 768

        self.dim = fea_dim
        self.dropout = dropout
        self.num_heads = 4

        # -------- Linear Projections --------
        self.linear_text = nn.Sequential(
            nn.Linear(self.text_dim, fea_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.linear_img = nn.Sequential(
            nn.Linear(self.img_dim, fea_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.linear_video = nn.Sequential(
            nn.Linear(self.video_dim, fea_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.linear_audio = nn.Sequential(
            nn.Linear(self.audio_dim, fea_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.linear_entity = nn.Sequential(
            nn.Linear(self.entity_dim, fea_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # -------- Cross-modal Interaction --------
        self.co_attention_ta = co_attention(
            d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads,
            dropout=dropout, d_model=fea_dim,
            visual_len=50, sen_len=512,
            fea_v=fea_dim, fea_s=fea_dim, pos=False
        )

        self.trm = nn.TransformerEncoderLayer(
            d_model=fea_dim,
            nhead=2,
            batch_first=True
        )

        # -------- Classifiers --------
        self.factual_classifier = nn.Linear(fea_dim, 2)
        self.counterfactual_classifier = nn.Linear(fea_dim, 2)
        self.entity_classifier = nn.Linear(fea_dim, 2)

        # -------- Fusion Hyper-parameter --------
        self.gamma = 0.1

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, **kwargs):

        # ===== Text =====
        title_inputid = kwargs['title_inputid']
        title_mask = kwargs['title_mask']

        text_feat = self.bert(
            title_inputid,
            attention_mask=title_mask
        ).last_hidden_state
        text_feat = self.linear_text(text_feat)
        text_feat = torch.mean(text_feat, dim=1)

        # ===== Image =====
        frames = kwargs['frames']
        img_feat = self.linear_img(frames)
        img_feat = torch.mean(img_feat, dim=1)

        # ===== Video =====
        c3d = kwargs['c3d']
        video_feat = self.linear_video(c3d)
        video_feat = torch.mean(video_feat, dim=1)

        # ===== Audio =====
        audio = kwargs['audio_feat']
        audio_feat = self.linear_audio(audio)
        audio_feat = torch.mean(audio_feat, dim=1)

        # ===== (Entity-level) =====
        entity_inputid = kwargs['entity_inputid']
        entity_mask = kwargs['entity_mask']

        entity_features = []
        for i in range(entity_inputid.size(0)):
            fea = self.bert(
                entity_inputid[i],
                attention_mask=entity_mask[i]
            ).pooler_output
            entity_features.append(fea)

        entity_feat = torch.stack(entity_features)
        entity_feat = self.linear_entity(entity_feat)

        # ===== Event-level Fusion =====
        fea_event = torch.stack(
            [text_feat, img_feat, video_feat, audio_feat],
            dim=1
        )
        fea_event = self.trm(fea_event)
        fea_event = torch.mean(fea_event, dim=1)

        # ===== Factual Prediction =====
        y_factual = self.factual_classifier(fea_event)

        # ===== Counterfactual Reasoning =====
        fea_cf = torch.stack(
            [text_feat, img_feat,
             video_feat.detach(),
             audio_feat.detach()],
            dim=1
        )
        fea_cf = self.trm(fea_cf)
        fea_cf = torch.mean(fea_cf, dim=1)

        y_counterfactual = self.counterfactual_classifier(fea_cf)

        # ===== Event-level Debiased Output =====
        y_reason = y_factual - y_counterfactual

        # ===== Entity-level Intervention =====
        y_entity = self.entity_classifier(entity_feat)

        # ===== Dual-view Fusion =====
        y_final = torch.sigmoid(
            self.gamma * y_reason + (1 - self.gamma) * y_entity
        )

        return {
            "y_pred": y_final,
            "y_factual": y_factual,
            "y_counterfactual": y_counterfactual,
            "y_entity": y_entity
        }
