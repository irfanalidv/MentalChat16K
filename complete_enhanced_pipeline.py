# Complete Enhanced Mental Health Research Framework - MentalChat16K Dataset
# GPU-Accelerated Pipeline with Real Data Integration and Paper Generation
# Version: 3.0 - Complete Implementation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import transformers
from transformers import AutoTokenizer, AutoModel, get_scheduler, AutoConfig
from torch.optim import AdamW
from datasets import load_dataset
import logging
import json
import os
from datetime import datetime
import warnings
import re
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import scipy.stats as stats
from scipy.stats import ttest_ind, chi2_contingency

warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
except:
    pass


class CompleteMentalChat16KPipeline:
    """Complete enhanced research pipeline with all functionality"""

    def __init__(self, gpu_enabled=True, experiment_name="complete_mentalchat16k"):
        self.device = torch.device(
            "cuda" if gpu_enabled and torch.cuda.is_available() else "cpu"
        )
        self.experiment_name = experiment_name
        self.results_dir = f"{experiment_name}_results"
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(
            max_features=10000, stop_words="english", ngram_range=(1, 2)
        )
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize metrics storage
        self.metrics = {}
        self.findings = []

        print("🚀 Complete MentalChat16K Research Pipeline Initialized")
        print(f"📱 Device: {self.device}")
        print(f"🔥 GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"📊 GPU: {torch.cuda.get_device_name()}")
            print(
                f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        print("=" * 60)

    def load_dataset(self):
        """Load and validate the MentalChat16K dataset"""
        self.logger.info("Loading MentalChat16K dataset")

        try:
            dataset = load_dataset("ShenLab/MentalChat16K")
            self.df = dataset["train"].to_pandas()

            print(f"📊 Dataset loaded successfully!")
            print(f"   • Total samples: {len(self.df)}")
            print(f"   • Columns: {list(self.df.columns)}")
            print(
                f"   • Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
            )

            # Display sample data
            print("\n🔍 Sample data:")
            for i in range(min(2, len(self.df))):
                print(f"\nSample {i+1}:")
                for col in self.df.columns:
                    value = str(self.df.iloc[i][col])
                    if len(value) > 100:
                        value = value[:100] + "..."
                    print(f"   {col}: {value}")

            self.metrics["dataset_loading"] = {
                "dataset_size": len(self.df),
                "columns": list(self.df.columns),
                "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
                "data_source": "ShenLab/MentalChat16K",
            }

            return self.df

        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            print(f"❌ Error loading dataset: {str(e)}")
            return None

    def comprehensive_analysis(self):
        """Perform comprehensive data analysis"""
        self.logger.info("Performing comprehensive data analysis")

        # Text length analysis
        if "input" in self.df.columns:
            self.df["input_length"] = self.df["input"].str.len()
            self.df["input_word_count"] = self.df["input"].str.split().str.len()

        if "output" in self.df.columns:
            self.df["output_length"] = self.df["output"].str.len()
            self.df["output_word_count"] = self.df["output"].str.split().str.len()

        # Sentiment analysis
        if "input" in self.df.columns:
            sentiments = []
            for text in self.df["input"].fillna(""):
                scores = self.sentiment_analyzer.polarity_scores(str(text))
                sentiments.append(scores["compound"])
            self.df["sentiment_score"] = sentiments

        # Create visualizations
        self._create_visualizations()

        # Statistical analysis
        analysis_stats = {
            "total_samples": len(self.df),
            "avg_input_length": (
                self.df["input_length"].mean()
                if "input_length" in self.df.columns
                else None
            ),
            "avg_output_length": (
                self.df["output_length"].mean()
                if "output_length" in self.df.columns
                else None
            ),
            "avg_sentiment": (
                self.df["sentiment_score"].mean()
                if "sentiment_score" in self.df.columns
                else None
            ),
            "sentiment_std": (
                self.df["sentiment_score"].std()
                if "sentiment_score" in self.df.columns
                else None
            ),
        }

        self.metrics["comprehensive_analysis"] = analysis_stats
        self.findings.append(
            f"Dataset contains {len(self.df)} conversation pairs with comprehensive analysis"
        )

        return analysis_stats

    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Text length distribution
        if "input_length" in self.df.columns:
            axes[0, 0].hist(
                self.df["input_length"].dropna(), bins=50, alpha=0.7, color="skyblue"
            )
            axes[0, 0].set_title("Input Text Length Distribution")
            axes[0, 0].set_xlabel("Character Count")
            axes[0, 0].set_ylabel("Frequency")

        if "output_length" in self.df.columns:
            axes[0, 1].hist(
                self.df["output_length"].dropna(),
                bins=50,
                alpha=0.7,
                color="lightcoral",
            )
            axes[0, 1].set_title("Output Text Length Distribution")
            axes[0, 1].set_xlabel("Character Count")
            axes[0, 1].set_ylabel("Frequency")

        # Sentiment analysis
        if "sentiment_score" in self.df.columns:
            axes[1, 0].hist(
                self.df["sentiment_score"], bins=30, alpha=0.7, color="lightgreen"
            )
            axes[1, 0].set_title("Sentiment Score Distribution")
            axes[1, 0].set_xlabel("Sentiment Score (-1 to 1)")
            axes[1, 0].set_ylabel("Frequency")

        # Word cloud
        if "input" in self.df.columns:
            all_text = " ".join(self.df["input"].fillna("").astype(str))
            all_text = re.sub(r"[^a-zA-Z\s]", "", all_text)
            wordcloud = WordCloud(
                width=400, height=300, background_color="white"
            ).generate(all_text)
            axes[1, 1].imshow(wordcloud, interpolation="bilinear")
            axes[1, 1].axis("off")
            axes[1, 1].set_title("Word Cloud - Input Text")

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/comprehensive_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def create_classification_tasks(self):
        """Create classification tasks"""
        self.logger.info("Creating classification tasks")

        # Sentiment-based classification
        if "sentiment_score" in self.df.columns:

            def categorize_sentiment(score):
                if score < -0.1:
                    return 0  # Negative/Distressed
                elif score > 0.1:
                    return 2  # Positive/Supportive
                else:
                    return 1  # Neutral

            self.df["sentiment_category"] = self.df["sentiment_score"].apply(
                categorize_sentiment
            )

        # Response type classification
        if "output" in self.df.columns:

            def classify_response_type(text):
                text = str(text).lower()
                if any(
                    word in text
                    for word in ["suggestion", "recommend", "try", "consider", "might"]
                ):
                    return 2  # Advice/Suggestions
                elif any(
                    word in text
                    for word in ["understand", "feel", "sorry", "empathy", "support"]
                ):
                    return 1  # Empathetic/Supportive
                else:
                    return 0  # Informational/General

            self.df["response_type"] = self.df["output"].apply(classify_response_type)

        # Print distributions
        print("📊 Classification Tasks Created:")
        for col in ["sentiment_category", "response_type"]:
            if col in self.df.columns:
                dist = self.df[col].value_counts().sort_index()
                print(f"   • {col.replace('_', ' ').title()}: {dict(dist)}")

        metrics = {
            "classification_tasks_created": sum(
                1
                for col in ["sentiment_category", "response_type"]
                if col in self.df.columns
            ),
            "sentiment_distribution": (
                self.df["sentiment_category"].value_counts().to_dict()
                if "sentiment_category" in self.df.columns
                else None
            ),
            "response_type_distribution": (
                self.df["response_type"].value_counts().to_dict()
                if "response_type" in self.df.columns
                else None
            ),
        }

        self.metrics["classification_creation"] = metrics
        return metrics

    def extract_features(self):
        """Extract advanced features"""
        self.logger.info("Extracting advanced features")

        # TF-IDF features
        if "input" in self.df.columns:
            tfidf_features = self.vectorizer.fit_transform(self.df["input"].fillna(""))
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_dense = tfidf_features.toarray()

            # Add top TF-IDF features
            top_features = np.argsort(np.mean(tfidf_dense, axis=0))[-10:]
            for i, feature_idx in enumerate(top_features):
                feature_name = feature_names[feature_idx].replace(" ", "_")
                self.df[f"tfidf_{feature_name}"] = tfidf_dense[:, feature_idx]

        # Psychological indicators
        if "input" in self.df.columns:
            stress_words = [
                "stress",
                "anxious",
                "worried",
                "overwhelmed",
                "pressure",
                "burden",
            ]
            positive_words = [
                "happy",
                "grateful",
                "hopeful",
                "better",
                "improving",
                "good",
            ]

            self.df["stress_indicators"] = self.df["input"].apply(
                lambda x: sum(1 for word in stress_words if word in str(x).lower())
            )
            self.df["positive_indicators"] = self.df["input"].apply(
                lambda x: sum(1 for word in positive_words if word in str(x).lower())
            )

        # Text complexity features
        if "input" in self.df.columns:
            self.df["word_count"] = self.df["input"].apply(
                lambda x: len(str(x).split())
            )
            self.df["sentence_count"] = self.df["input"].apply(
                lambda x: len(str(x).split("."))
            )
            self.df["avg_word_length"] = self.df["input"].apply(
                lambda x: (
                    np.mean([len(word) for word in str(x).split()])
                    if str(x).split()
                    else 0
                )
            )

        # Feature summary
        feature_cols = [
            col
            for col in self.df.columns
            if any(
                prefix in col
                for prefix in [
                    "tfidf_",
                    "stress_",
                    "positive_",
                    "word_",
                    "sentence_",
                    "avg_",
                ]
            )
        ]

        print(f"🎯 Features Extracted: {len(feature_cols)}")

        metrics = {
            "total_features_extracted": len(feature_cols),
            "feature_categories": {
                "tfidf": sum(1 for col in feature_cols if "tfidf_" in col),
                "psychological": sum(
                    1
                    for col in feature_cols
                    if any(x in col for x in ["stress_", "positive_"])
                ),
                "linguistic": sum(
                    1
                    for col in feature_cols
                    if any(x in col for x in ["word_", "sentence_", "avg_"])
                ),
            },
        }

        self.metrics["feature_extraction"] = metrics
        return feature_cols

    def train_bert_classifier(
        self, target_column="sentiment_category", epochs=3, batch_size=16
    ):
        """Train BERT classifier"""
        self.logger.info(f"Training BERT classifier for {target_column}")

        if target_column not in self.df.columns:
            print(f"❌ Target column '{target_column}' not found!")
            return None, None

        # Prepare data
        texts = self.df["input"].fillna("")
        labels = self.df[target_column]

        # Remove NaN labels
        valid_mask = ~labels.isna()
        texts = texts[valid_mask]
        labels = labels[valid_mask]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        num_classes = len(labels.unique())

        # Simple BERT classifier
        class SimpleBERTClassifier(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.bert = AutoModel.from_pretrained("bert-base-uncased")
                self.dropout = nn.Dropout(0.3)
                self.classifier = nn.Linear(768, num_classes)

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                output = self.dropout(pooled_output)
                return self.classifier(output)

        model = SimpleBERTClassifier(num_classes).to(self.device)

        # Create datasets
        class SimpleDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = str(
                    self.texts.iloc[idx]
                    if hasattr(self.texts, "iloc")
                    else self.texts[idx]
                )
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                    "labels": torch.tensor(
                        (
                            self.labels.iloc[idx]
                            if hasattr(self.labels, "iloc")
                            else self.labels[idx]
                        ),
                        dtype=torch.long,
                    ),
                }

        train_dataset = SimpleDataset(X_train, y_train, tokenizer)
        test_dataset = SimpleDataset(X_test, y_test, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Training setup
        optimizer = AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        training_losses = []

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_batch = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 20 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                    )

            avg_loss = total_loss / len(train_loader)
            training_losses.append(avg_loss)
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_batch = batch["labels"].to(self.device)

                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average="weighted")

        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(training_losses, "b-", linewidth=2)
        plt.title(f"BERT Training Loss - {target_column}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(
            f"{self.results_dir}/bert_training_curve_{target_column}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        metrics = {
            "model_type": "BERT",
            "target_column": target_column,
            "accuracy": accuracy,
            "f1_score": f1,
            "epochs": epochs,
            "final_loss": training_losses[-1],
            "num_classes": num_classes,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
        }

        self.metrics[f"bert_training_{target_column}"] = metrics
        self.findings.append(
            f"BERT classifier achieved {accuracy:.3f} accuracy on {target_column} prediction"
        )

        return model, metrics

    def train_feature_model(self, target_column="response_type"):
        """Train feature-based model"""
        self.logger.info(f"Training feature-based model for {target_column}")

        if target_column not in self.df.columns:
            print(f"❌ Target column '{target_column}' not found!")
            return None, None

        # Select features
        feature_cols = [
            col
            for col in self.df.columns
            if any(
                prefix in col
                for prefix in [
                    "tfidf_",
                    "stress_",
                    "positive_",
                    "word_",
                    "sentence_",
                    "avg_",
                    "sentiment_score",
                    "input_length",
                    "output_length",
                ]
            )
        ]

        X = self.df[feature_cols].fillna(0)
        y = self.df[target_column].fillna(0)

        # Remove samples with NaN labels
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Simple neural network
        class SimpleFeatureModel(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )

            def forward(self, x):
                return self.network(x)

        num_classes = len(y.unique())
        model = SimpleFeatureModel(X_train_scaled.shape[1], num_classes).to(self.device)

        # Create datasets
        class FeatureDataset(Dataset):
            def __init__(self, features, labels):
                self.features = features
                self.labels = labels

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return {
                    "features": torch.tensor(self.features[idx], dtype=torch.float32),
                    "labels": torch.tensor(
                        (
                            self.labels.iloc[idx]
                            if hasattr(self.labels, "iloc")
                            else self.labels[idx]
                        ),
                        dtype=torch.long,
                    ),
                }

        train_dataset = FeatureDataset(X_train_scaled, y_train)
        test_dataset = FeatureDataset(X_test_scaled, y_test)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 50
        train_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            # Training
            model.train()
            total_loss = 0
            for batch in train_loader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Validation
            model.eval()
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch in test_loader:
                    features = batch["features"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = model(features)
                    predictions = torch.argmax(outputs, dim=1)

                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            val_accuracy = accuracy_score(val_targets, val_predictions)
            avg_loss = total_loss / len(train_loader)

            train_losses.append(avg_loss)
            val_accuracies.append(val_accuracy)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Loss = {avg_loss:.4f}, Val Accuracy = {val_accuracy:.4f}"
                )

        # Final evaluation
        f1 = f1_score(val_targets, val_predictions, average="weighted")

        # Plot training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(train_losses, "b-", label="Training Loss")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(val_accuracies, "r-", label="Validation Accuracy")
        ax2.set_title("Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/training_curves_{target_column}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        metrics = {
            "model_type": "Feature-based Neural Network",
            "target_column": target_column,
            "accuracy": val_accuracy,
            "f1_score": f1,
            "final_loss": train_losses[-1],
            "num_features": len(feature_cols),
            "num_classes": num_classes,
            "epochs": num_epochs,
        }

        self.metrics[f"feature_model_{target_column}"] = metrics
        self.findings.append(
            f"Feature-based model achieved {val_accuracy:.3f} accuracy on {target_column}"
        )

        return model, metrics

    def cross_validation_analysis(self, target_column="sentiment_category", cv_folds=5):
        """Perform cross-validation analysis"""
        self.logger.info(f"Performing {cv_folds}-fold cross-validation")

        # Select features
        feature_cols = [
            col
            for col in self.df.columns
            if any(
                prefix in col
                for prefix in [
                    "tfidf_",
                    "stress_",
                    "positive_",
                    "word_",
                    "sentence_",
                    "avg_",
                    "sentiment_score",
                    "input_length",
                    "output_length",
                ]
            )
        ]

        X = self.df[feature_cols].fillna(0)
        y = self.df[target_column].fillna(0)

        # Remove samples with NaN labels
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        cv_f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Training fold {fold + 1}/{cv_folds}...")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)

            # Train simple model
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train_fold)

            # Evaluate
            predictions = model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val_fold, predictions)
            f1 = f1_score(y_val_fold, predictions, average="weighted")

            cv_scores.append(accuracy)
            cv_f1_scores.append(f1)

            print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # CV Results
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)
        mean_f1 = np.mean(cv_f1_scores)
        std_f1 = np.std(cv_f1_scores)

        print(f"\n📊 Cross-Validation Results ({cv_folds}-fold):")
        print(f"   • Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"   • F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")

        # Plot CV results
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(1, cv_folds + 1), cv_scores, alpha=0.7, color="skyblue")
        plt.axhline(
            y=mean_accuracy,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_accuracy:.4f}",
        )
        plt.title("Cross-Validation Accuracy Scores")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.bar(range(1, cv_folds + 1), cv_f1_scores, alpha=0.7, color="lightcoral")
        plt.axhline(
            y=mean_f1, color="red", linestyle="--", label=f"Mean: {mean_f1:.4f}"
        )
        plt.title("Cross-Validation F1 Scores")
        plt.xlabel("Fold")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/cv_results_{target_column}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        metrics = {
            "cv_folds": cv_folds,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "individual_scores": {"accuracy": cv_scores, "f1": cv_f1_scores},
        }

        self.metrics["cross_validation"] = metrics
        self.findings.append(
            f"Cross-validation shows {mean_accuracy:.3f} ± {std_accuracy:.3f} accuracy"
        )

        return metrics

    def generate_paper_metrics(self):
        """Generate metrics for paper integration"""
        self.logger.info("Generating paper-ready metrics")

        # Compile all results
        paper_metrics = {
            "abstract_metrics": {},
            "results_tables": {},
            "dataset_statistics": {},
            "model_performance": {},
        }

        # Extract key metrics
        for stage, metrics in self.metrics.items():
            if "accuracy" in metrics:
                paper_metrics["abstract_metrics"][f"{stage}_accuracy"] = round(
                    metrics["accuracy"], 4
                )
            if "f1_score" in metrics:
                paper_metrics["abstract_metrics"][f"{stage}_f1"] = round(
                    metrics["f1_score"], 4
                )

        # Generate LaTeX table data
        paper_metrics["latex_tables"] = self._generate_latex_tables()

        # Save paper metrics
        with open(f"{self.results_dir}/paper_metrics.json", "w") as f:
            json.dump(paper_metrics, f, indent=2)

        print("📝 Paper-ready metrics generated!")
        print(f"   • Abstract metrics: {len(paper_metrics['abstract_metrics'])}")
        print(f"   • LaTeX tables: {len(paper_metrics['latex_tables'])}")
        print(f"   • Saved to: {self.results_dir}/paper_metrics.json")

        return paper_metrics

    def _generate_latex_tables(self):
        """Generate LaTeX formatted tables"""
        latex_tables = {}

        # Model performance table
        table_data = []
        for stage, metrics in self.metrics.items():
            if "accuracy" in metrics and "f1_score" in metrics:
                model_name = stage.replace("_", " ").title()
                accuracy = f"{metrics['accuracy']:.3f}"
                f1 = f"{metrics['f1_score']:.3f}"
                table_data.append([model_name, accuracy, f1])

        latex_tables["model_performance"] = table_data

        # Dataset statistics
        if "dataset_loading" in self.metrics:
            metrics = self.metrics["dataset_loading"]
            latex_tables["dataset_stats"] = {
                "total_samples": metrics.get("dataset_size", 0),
                "columns": len(metrics.get("columns", [])),
                "memory_usage": f"{metrics.get('memory_usage_mb', 0):.1f} MB",
            }

        return latex_tables

    def run_complete_pipeline(self):
        """Execute the complete research pipeline"""
        print("🚀 COMPLETE MENTALCHAT16K RESEARCH PIPELINE")
        print("=" * 70)
        print("GPU-accelerated analysis with real data integration")
        print("Comprehensive statistical validation and paper-ready metrics")
        print("=" * 70)

        # Stage 1: Load dataset
        print("\n📥 STAGE 1: Loading MentalChat16K Dataset")
        df = self.load_dataset()
        if df is None:
            print("❌ Failed to load dataset. Please check your internet connection.")
            return None, None

        # Stage 2: Comprehensive analysis
        print("\n🔍 STAGE 2: Comprehensive Data Analysis")
        analysis_results = self.comprehensive_analysis()

        # Stage 3: Create classification tasks
        print("\n🎯 STAGE 3: Creating Classification Tasks")
        classification_metrics = self.create_classification_tasks()

        # Stage 4: Extract features
        print("\n⚙️ STAGE 4: Feature Extraction")
        feature_cols = self.extract_features()

        # Stage 5: Train BERT classifier
        print("\n🤖 STAGE 5: Training BERT Classifier")
        bert_model, bert_metrics = self.train_bert_classifier(
            target_column="sentiment_category", epochs=3, batch_size=16
        )

        # Stage 6: Train feature-based model
        print("\n🧠 STAGE 6: Training Feature-Based Model")
        feature_model, feature_metrics = self.train_feature_model(
            target_column="response_type"
        )

        # Stage 7: Cross-validation analysis
        print("\n📊 STAGE 7: Cross-Validation Analysis")
        cv_metrics = self.cross_validation_analysis(
            target_column="sentiment_category", cv_folds=5
        )

        # Stage 8: Generate paper metrics
        print("\n📝 STAGE 8: Generating Paper-Ready Metrics")
        paper_metrics = self.generate_paper_metrics()

        print("\n✅ Complete research pipeline completed successfully!")
        print(f"📁 All results saved to: {self.results_dir}/")
        print(f"📊 Paper metrics: paper_metrics.json")

        return self, paper_metrics


def run_complete_mentalchat16k_research():
    """Execute the complete research pipeline"""
    pipeline = CompleteMentalChat16KPipeline(gpu_enabled=True)
    return pipeline.run_complete_pipeline()


if __name__ == "__main__":
    print("🚀 Starting Complete MentalChat16K Research Pipeline")
    print("=" * 60)

    # Run the complete pipeline
    pipeline, results = run_complete_mentalchat16k_research()
