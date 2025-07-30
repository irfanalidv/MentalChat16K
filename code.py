# Mental Health Research Framework - MentalChat16K Dataset
# GPU-Accelerated Pipeline for Conversational Mental Health Analysis
# Version: 2.0 - Real Data Implementation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import transformers
from transformers import AutoTokenizer, AutoModel, get_scheduler
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
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class MentalHealthResearchTracker:
    """Advanced status tracking system for mental health research pipeline"""

    def __init__(self, experiment_name="mentalchat16k_research"):
        self.experiment_name = experiment_name
        self.status_file = f"{experiment_name}_status.json"
        self.log_file = f"{experiment_name}_log.txt"
        self.results_dir = f"{experiment_name}_results"
        self.initialize_logging()
        self.status = self.load_status()
        os.makedirs(self.results_dir, exist_ok=True)

    def initialize_logging(self):
        """Initialize comprehensive logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_status(self):
        """Load previous experiment status"""
        if os.path.exists(self.status_file):
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {
            "experiment_name": self.experiment_name,
            "dataset_source": "ShenLab/MentalChat16K",
            "created": datetime.now().isoformat(),
            "stages_completed": [],
            "current_stage": None,
            "metrics": {},
            "model_checkpoints": [],
            "research_findings": [],
            "last_updated": datetime.now().isoformat()
        }

    def save_status(self):
        """Save current status to file"""
        self.status["last_updated"] = datetime.now().isoformat()
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def update_stage(self, stage_name, metrics=None, findings=None):
        """Update current research stage with metrics and findings"""
        self.status["current_stage"] = stage_name
        if stage_name not in self.status["stages_completed"]:
            self.status["stages_completed"].append(stage_name)

        if metrics:
            self.status["metrics"][stage_name] = metrics

        if findings:
            self.status["research_findings"].extend(findings)

        self.save_status()
        self.logger.info(f"Stage completed: {stage_name}")

        # Print stage completion
        print(f"✅ {stage_name.replace('_', ' ').title()} - Complete")
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   📊 {key}: {value:.4f}" if isinstance(value, float) else f"   📊 {key}: {value}")

class MentalChatDataset(Dataset):
    """Custom dataset for MentalChat16K data with both text and numerical features"""

    def __init__(self, texts, labels, tokenizer=None, max_length=512, mode='text'):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.mode == 'text' and self.tokenizer:
            text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        else:
            # For feature-based models
            return {
                'features': torch.tensor(self.texts[idx], dtype=torch.float32),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

class MentalHealthBERTClassifier(nn.Module):
    """Advanced BERT-based classifier for mental health conversation analysis"""

    def __init__(self, model_name='bert-base-uncased', num_classes=3, dropout_rate=0.3):
        super(MentalHealthBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class AdvancedMentalHealthPredictor(nn.Module):
    """Advanced neural network for mental health prediction with attention mechanism"""

    def __init__(self, input_dim, hidden_dims=[512, 256, 128], num_classes=3, dropout_rate=0.3):
        super(AdvancedMentalHealthPredictor, self).__init__()

        # Feature attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Softmax(dim=1)
        )

        # Main network
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate * (0.8 ** i))  # Decreasing dropout
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Apply attention
        attention_weights = self.attention(x)
        x_attended = x * attention_weights
        return self.network(x_attended)

class MentalChat16KResearchPipeline:
    """Complete research pipeline for MentalChat16K analysis"""

    def __init__(self, gpu_enabled=True):
        self.device = torch.device('cuda' if gpu_enabled and torch.cuda.is_available() else 'cpu')
        self.tracker = MentalHealthResearchTracker()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        print("🚀 MentalChat16K Research Pipeline Initialized")
        print(f"📱 Device: {self.device}")
        print(f"🔥 GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"📊 GPU: {torch.cuda.get_device_name()}")
        print("="*60)

    def load_mentalchat16k_dataset(self):
        """Load the actual MentalChat16K dataset from Hugging Face"""
        self.tracker.logger.info("Loading MentalChat16K dataset from Hugging Face")

        try:
            # Load the dataset
            dataset = load_dataset("ShenLab/MentalChat16K")

            # Convert to pandas DataFrame
            train_data = dataset['train'].to_pandas()

            self.df = train_data.copy()

            print(f"📊 Dataset loaded successfully!")
            print(f"   • Total samples: {len(self.df)}")
            print(f"   • Columns: {list(self.df.columns)}")
            print(f"   • Data types: {dict(self.df.dtypes)}")

            # Display sample data
            print("\n🔍 Sample data:")
            for i in range(min(2, len(self.df))):
                print(f"\nSample {i+1}:")
                for col in self.df.columns:
                    value = str(self.df.iloc[i][col])[:100] + "..." if len(str(self.df.iloc[i][col])) > 100 else str(self.df.iloc[i][col])
                    print(f"   {col}: {value}")

            metrics = {
                "dataset_size": len(self.df),
                "columns": list(self.df.columns),
                "missing_values": self.df.isnull().sum().to_dict(),
                "data_source": "ShenLab/MentalChat16K"
            }

            self.tracker.update_stage("dataset_loading", metrics)
            return self.df

        except Exception as e:
            self.tracker.logger.error(f"Error loading dataset: {str(e)}")
            print(f"❌ Error loading dataset: {str(e)}")
            return None

    def analyze_dataset_characteristics(self):
        """Comprehensive analysis of the MentalChat16K dataset"""
        self.tracker.logger.info("Analyzing dataset characteristics")

        # Text length analysis
        if 'input' in self.df.columns:
            self.df['input_length'] = self.df['input'].str.len()
        if 'output' in self.df.columns:
            self.df['output_length'] = self.df['output'].str.len()

        # Sentiment analysis
        if 'input' in self.df.columns:
            sentiments = []
            for text in self.df['input'].fillna(''):
                scores = self.sentiment_analyzer.polarity_scores(str(text))
                sentiments.append(scores['compound'])
            self.df['sentiment_score'] = sentiments

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Text length distribution
        if 'input_length' in self.df.columns:
            axes[0,0].hist(self.df['input_length'].dropna(), bins=50, alpha=0.7, color='skyblue')
            axes[0,0].set_title('Distribution of Input Text Lengths')
            axes[0,0].set_xlabel('Character Count')
            axes[0,0].set_ylabel('Frequency')

        if 'output_length' in self.df.columns:
            axes[0,1].hist(self.df['output_length'].dropna(), bins=50, alpha=0.7, color='lightcoral')
            axes[0,1].set_title('Distribution of Output Text Lengths')
            axes[0,1].set_xlabel('Character Count')
            axes[0,1].set_ylabel('Frequency')

        # Sentiment analysis
        if 'sentiment_score' in self.df.columns:
            axes[1,0].hist(self.df['sentiment_score'], bins=30, alpha=0.7, color='lightgreen')
            axes[1,0].set_title('Sentiment Score Distribution')
            axes[1,0].set_xlabel('Sentiment Score (-1 to 1)')
            axes[1,0].set_ylabel('Frequency')

        # Word cloud of input text
        if 'input' in self.df.columns:
            all_text = ' '.join(self.df['input'].fillna('').astype(str))
            # Clean text
            all_text = re.sub(r'[^a-zA-Z\s]', '', all_text)
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_text)
            axes[1,1].imshow(wordcloud, interpolation='bilinear')
            axes[1,1].axis('off')
            axes[1,1].set_title('Word Cloud of Input Text')

        plt.tight_layout()
        plt.savefig(f'{self.tracker.results_dir}/dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Statistical summary
        analysis_stats = {
            "total_samples": len(self.df),
            "avg_input_length": self.df['input_length'].mean() if 'input_length' in self.df.columns else None,
            "avg_output_length": self.df['output_length'].mean() if 'output_length' in self.df.columns else None,
            "avg_sentiment": self.df['sentiment_score'].mean() if 'sentiment_score' in self.df.columns else None,
            "sentiment_std": self.df['sentiment_score'].std() if 'sentiment_score' in self.df.columns else None
        }

        findings = [
            f"Dataset contains {len(self.df)} conversation pairs",
            f"Average input length: {analysis_stats['avg_input_length']:.0f} characters" if analysis_stats['avg_input_length'] else "Input length analysis unavailable",
            f"Average output length: {analysis_stats['avg_output_length']:.0f} characters" if analysis_stats['avg_output_length'] else "Output length analysis unavailable",
            f"Overall sentiment tends toward neutral with slight positive bias" if analysis_stats['avg_sentiment'] and analysis_stats['avg_sentiment'] > 0 else "Sentiment analysis completed"
        ]

        self.tracker.update_stage("dataset_analysis", analysis_stats, findings)
        return analysis_stats

    def create_classification_task(self):
        """Create classification labels for the mental health conversations"""
        self.tracker.logger.info("Creating classification task from conversation data")

        # Create sentiment-based classification
        if 'sentiment_score' in self.df.columns:
            def categorize_sentiment(score):
                if score < -0.1:
                    return 0  # Negative/Distressed
                elif score > 0.1:
                    return 2  # Positive/Supportive
                else:
                    return 1  # Neutral

            self.df['sentiment_category'] = self.df['sentiment_score'].apply(categorize_sentiment)

        # Create response type classification based on output characteristics
        if 'output' in self.df.columns:
            def classify_response_type(text):
                text = str(text).lower()
                if any(word in text for word in ['suggestion', 'recommend', 'try', 'consider', 'might']):
                    return 2  # Advice/Suggestions
                elif any(word in text for word in ['understand', 'feel', 'sorry', 'empathy', 'support']):
                    return 1  # Empathetic/Supportive
                else:
                    return 0  # Informational/General

            self.df['response_type'] = self.df['output'].apply(classify_response_type)

        # Create conversation length-based classification
        if 'output_length' in self.df.columns:
            def classify_response_length(length):
                if length < 100:
                    return 0  # Short response
                elif length < 300:
                    return 1  # Medium response
                else:
                    return 2  # Long response

            self.df['response_length_category'] = self.df['output_length'].apply(classify_response_length)

        # Print classification distributions
        print("📊 Classification Task Created:")
        for col in ['sentiment_category', 'response_type', 'response_length_category']:
            if col in self.df.columns:
                dist = self.df[col].value_counts().sort_index()
                print(f"   • {col.replace('_', ' ').title()}: {dict(dist)}")

        metrics = {
            "classification_tasks_created": sum(1 for col in ['sentiment_category', 'response_type', 'response_length_category'] if col in self.df.columns),
            "sentiment_distribution": self.df['sentiment_category'].value_counts().to_dict() if 'sentiment_category' in self.df.columns else None,
            "response_type_distribution": self.df['response_type'].value_counts().to_dict() if 'response_type' in self.df.columns else None
        }

        self.tracker.update_stage("classification_creation", metrics)
        return metrics

    def extract_advanced_features(self):
        """Extract advanced linguistic and psychological features"""
        self.tracker.logger.info("Extracting advanced features for analysis")

        # TF-IDF features
        if 'input' in self.df.columns:
            tfidf_features = self.vectorizer.fit_transform(self.df['input'].fillna(''))

            # Get top TF-IDF features
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_dense = tfidf_features.toarray()

            # Add selected TF-IDF features to dataframe
            top_features = np.argsort(np.mean(tfidf_dense, axis=0))[-50:]  # Top 50 features
            for i, feature_idx in enumerate(top_features[-10:]):  # Use top 10 for the model
                self.df[f'tfidf_{feature_names[feature_idx]}'] = tfidf_dense[:, feature_idx]

        # Psychological indicators
        if 'input' in self.df.columns:
            # Count psychological keywords
            stress_words = ['stress', 'anxious', 'worried', 'overwhelmed', 'pressure', 'burden']
            positive_words = ['happy', 'grateful', 'hopeful', 'better', 'improving', 'good']

            self.df['stress_indicators'] = self.df['input'].apply(
                lambda x: sum(1 for word in stress_words if word in str(x).lower())
            )
            self.df['positive_indicators'] = self.df['input'].apply(
                lambda x: sum(1 for word in positive_words if word in str(x).lower())
            )

        # Text complexity features
        if 'input' in self.df.columns:
            self.df['word_count'] = self.df['input'].apply(lambda x: len(str(x).split()))
            self.df['sentence_count'] = self.df['input'].apply(lambda x: len(str(x).split('.')))
            self.df['avg_word_length'] = self.df['input'].apply(
                lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0
            )

        # Feature summary
        feature_cols = [col for col in self.df.columns if any(prefix in col for prefix in ['tfidf_', 'stress_', 'positive_', 'word_', 'sentence_', 'avg_'])]

        print(f"🎯 Advanced Features Extracted:")
        print(f"   • Total features: {len(feature_cols)}")
        print(f"   • TF-IDF features: {sum(1 for col in feature_cols if 'tfidf_' in col)}")
        print(f"   • Psychological features: {sum(1 for col in feature_cols if any(x in col for x in ['stress_', 'positive_']))}")
        print(f"   • Linguistic features: {sum(1 for col in feature_cols if any(x in col for x in ['word_', 'sentence_', 'avg_']))}")

        metrics = {
            "total_features_extracted": len(feature_cols),
            "feature_categories": {
                "tfidf": sum(1 for col in feature_cols if 'tfidf_' in col),
                "psychological": sum(1 for col in feature_cols if any(x in col for x in ['stress_', 'positive_'])),
                "linguistic": sum(1 for col in feature_cols if any(x in col for x in ['word_', 'sentence_', 'avg_']))
            }
        }

        self.tracker.update_stage("feature_extraction", metrics)
        return feature_cols

    def train_bert_classifier(self, target_column='sentiment_category', epochs=3, batch_size=16):
        """Train BERT classifier on the MentalChat16K data"""
        self.tracker.logger.info(f"Training BERT classifier for {target_column}")

        if target_column not in self.df.columns:
            print(f"❌ Target column '{target_column}' not found!")
            return None, None

        # Prepare data
        texts = self.df['input'].fillna('')
        labels = self.df[target_column]

        # Remove any NaN labels
        valid_mask = ~labels.isna()
        texts = texts[valid_mask]
        labels = labels[valid_mask]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        num_classes = len(labels.unique())
        model = MentalHealthBERTClassifier(num_classes=num_classes).to(self.device)

        # Create datasets and dataloaders
        train_dataset = MentalChatDataset(X_train, y_train.values, tokenizer=tokenizer)
        test_dataset = MentalChatDataset(X_test, y_test.values, tokenizer=tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Training setup
        optimizer = AdamW(model.parameters(), lr=2e-5)
        num_training_steps = epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        training_losses = []

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                total_loss += loss.item()

                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            training_losses.append(avg_loss)
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)

                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        class_report = classification_report(all_labels, all_predictions, output_dict=True)

        # Save model
        model_path = f"{self.tracker.results_dir}/bert_classifier_{target_column}.pth"
        torch.save(model.state_dict(), model_path)

        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(training_losses, 'b-', linewidth=2)
        plt.title(f'BERT Training Loss - {target_column}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'{self.tracker.results_dir}/bert_training_curve_{target_column}.png', dpi=300, bbox_inches='tight')
        plt.show()

        metrics = {
            "model_type": "BERT",
            "target_column": target_column,
            "accuracy": accuracy,
            "f1_score": f1,
            "epochs": epochs,
            "final_loss": training_losses[-1],
            "model_path": model_path,
            "num_classes": num_classes,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }

        findings = [
            f"BERT classifier achieved {accuracy:.3f} accuracy on {target_column} prediction",
            f"F1-score: {f1:.3f} indicating {('excellent' if f1 > 0.8 else 'good' if f1 > 0.6 else 'moderate')} performance",
            f"Model successfully trained on {len(X_train)} samples with {num_classes} classes"
        ]

        self.tracker.update_stage(f"bert_training_{target_column}", metrics, findings)
        return model, metrics

    def train_feature_based_model(self, target_column='response_type'):
        """Train advanced neural network on extracted features"""
        self.tracker.logger.info(f"Training feature-based model for {target_column}")

        if target_column not in self.df.columns:
            print(f"❌ Target column '{target_column}' not found!")
            return None, None

        # Select numerical features
        feature_cols = [col for col in self.df.columns if any(prefix in col for prefix in
                       ['tfidf_', 'stress_', 'positive_', 'word_', 'sentence_', 'avg_', 'sentiment_score', 'input_length', 'output_length'])]

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

        # Create datasets
        train_dataset = MentalChatDataset(X_train_scaled, y_train.values, mode='features')
        test_dataset = MentalChatDataset(X_test_scaled, y_test.values, mode='features')

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Initialize model
        num_classes = len(y.unique())
        model = AdvancedMentalHealthPredictor(
            input_dim=X_train_scaled.shape[1],
            num_classes=num_classes
        ).to(self.device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Training loop
        num_epochs = 50
        train_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            # Training
            model.train()
            total_loss = 0
            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)

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
                    features = batch['features'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = model(features)
                    predictions = torch.argmax(outputs, dim=1)

                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            val_accuracy = accuracy_score(val_targets, val_predictions)
            avg_loss = total_loss / len(train_loader)

            train_losses.append(avg_loss)
            val_accuracies.append(val_accuracy)

            scheduler.step(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")

        # Final evaluation
        f1 = f1_score(val_targets, val_predictions, average='weighted')

        # Feature importance analysis
        sample_input = torch.randn(1, X_train_scaled.shape[1]).to(self.device)
        sample_input.requires_grad_(True)
        output = model(sample_input)
        output.sum().backward()
        importance_scores = torch.abs(sample_input.grad).cpu().numpy().flatten()

        # Create feature importance plot
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Feature Importance - {target_column}')
        plt.tight_layout()
        plt.savefig(f'{self.tracker.results_dir}/feature_importance_{target_column}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Plot training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(train_losses, 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'{self.tracker.results_dir}/training_curves_{target_column}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Save model
        model_path = f"{self.tracker.results_dir}/feature_model_{target_column}.pth"
        torch.save(model.state_dict(), model_path)

        metrics = {
            "model_type": "Feature-based Neural Network",
            "target_column": target_column,
            "accuracy": val_accuracy,
            "f1_score": f1,
            "final_loss": train_losses[-1],
            "num_features": len(feature_cols),
            "num_classes": num_classes,
            "epochs": num_epochs,
            "model_path": model_path,
            "top_features": importance_df.head(10)['feature'].tolist()
        }

        findings = [
            f"Feature-based model achieved {val_accuracy:.3f} accuracy on {target_column}",
            f"Top predictive features: {', '.join(importance_df.head(3)['feature'].tolist())}",
            f"Model successfully utilized {len(feature_cols)} engineered features"
        ]

        self.tracker.update_stage(f"feature_model_{target_column}", metrics, findings)
        return model, metrics

    def cross_validation_analysis(self, target_column='sentiment_category', cv_folds=5):
        """Perform cross-validation analysis for robust evaluation"""
        self.tracker.logger.info(f"Performing {cv_folds}-fold cross-validation")

        # Select features and target
        feature_cols = [col for col in self.df.columns if any(prefix in col for prefix in
                       ['tfidf_', 'stress_', 'positive_', 'word_', 'sentence_', 'avg_', 'sentiment_score', 'input_length', 'output_length'])]

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

            # Train model
            num_classes = len(y.unique())
            model = AdvancedMentalHealthPredictor(
                input_dim=X_train_scaled.shape[1],
                num_classes=num_classes
            ).to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001)

            # Quick training (fewer epochs for CV)
            train_dataset = MentalChatDataset(X_train_scaled, y_train_fold.values, mode='features')
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            for epoch in range(10):  # Reduced epochs for CV
                model.train()
                for batch in train_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                val_features = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
                outputs = model(val_features)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            accuracy = accuracy_score(y_val_fold, predictions)
            f1 = f1_score(y_val_fold, predictions, average='weighted')

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
        plt.bar(range(1, cv_folds + 1), cv_scores, alpha=0.7, color='skyblue')
        plt.axhline(y=mean_accuracy, color='red', linestyle='--', label=f'Mean: {mean_accuracy:.4f}')
        plt.title('Cross-Validation Accuracy Scores')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.bar(range(1, cv_folds + 1), cv_f1_scores, alpha=0.7, color='lightcoral')
        plt.axhline(y=mean_f1, color='red', linestyle='--', label=f'Mean: {mean_f1:.4f}')
        plt.title('Cross-Validation F1 Scores')
        plt.xlabel('Fold')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.tracker.results_dir}/cv_results_{target_column}.png', dpi=300, bbox_inches='tight')
        plt.show()

        metrics = {
            "cv_folds": cv_folds,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "individual_scores": {
                "accuracy": cv_scores,
                "f1": cv_f1_scores
            }
        }

        findings = [
            f"Cross-validation shows consistent performance with {mean_accuracy:.3f} ± {std_accuracy:.3f} accuracy",
            f"Model generalization is {'excellent' if std_accuracy < 0.05 else 'good' if std_accuracy < 0.1 else 'moderate'}",
            f"F1-score stability: {mean_f1:.3f} ± {std_f1:.3f}"
        ]

        self.tracker.update_stage("cross_validation", metrics, findings)
        return metrics

    def generate_research_report(self):
        """Generate comprehensive research report for publication"""
        self.tracker.logger.info("Generating comprehensive research report")

        # Compile all results
        report = {
            "title": "Conversational Mental Health Analysis Using MentalChat16K Dataset",
            "abstract": {
                "dataset": "MentalChat16K - 16,113 mental health conversation pairs",
                "methods": ["BERT-based text classification", "Feature-based neural networks", "Cross-validation analysis"],
                "key_findings": self.tracker.status["research_findings"]
            },
            "methodology": {
                "dataset_source": "ShenLab/MentalChat16K (Hugging Face)",
                "preprocessing": ["Sentiment analysis", "Feature extraction", "Text vectorization"],
                "models": ["BERT classifier", "Advanced neural network with attention"],
                "evaluation": ["Accuracy", "F1-score", "Cross-validation", "Feature importance"]
            },
            "results": self.tracker.status["metrics"],
            "stages_completed": self.tracker.status["stages_completed"],
            "total_runtime": datetime.now().isoformat(),
            "gpu_utilized": str(self.device),
            "files_generated": [
                f"{self.tracker.results_dir}/dataset_analysis.png",
                f"{self.tracker.results_dir}/feature_importance_*.png",
                f"{self.tracker.results_dir}/training_curves_*.png",
                f"{self.tracker.results_dir}/cv_results_*.png"
            ]
        }

        # Save comprehensive report
        with open(f'{self.tracker.results_dir}/research_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate publication-ready summary
        print("📝 RESEARCH REPORT SUMMARY")
        print("="*60)
        print(f"📊 Dataset: {report['abstract']['dataset']}")
        print(f"🔬 Models Trained: {len([s for s in self.tracker.status['stages_completed'] if 'training' in s])}")
        print(f"📈 Analysis Stages: {len(self.tracker.status['stages_completed'])}")
        print(f"🎯 Key Findings: {len(self.tracker.status['research_findings'])}")

        print("\n🏆 MODEL PERFORMANCE SUMMARY:")
        for stage, metrics in self.tracker.status["metrics"].items():
            if "accuracy" in metrics and "f1_score" in metrics:
                print(f"   • {stage.replace('_', ' ').title()}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")

        print(f"\n📁 Results saved to: {self.tracker.results_dir}/")
        print(f"📋 Full report: research_report.json")
        print(f"📊 Status tracking: {self.tracker.status_file}")

        return report

def run_complete_mentalchat16k_research():
    """Execute the complete research pipeline using MentalChat16K"""
    print("🚀 MENTALCHAT16K RESEARCH PIPELINE")
    print("="*70)
    print("Using REAL data from University of Pennsylvania's MentalChat16K dataset")
    print("GPU-accelerated neural networks for mental health conversation analysis")
    print("="*70)

    # Initialize pipeline
    pipeline = MentalChat16KResearchPipeline(gpu_enabled=True)

    # Stage 1: Load real dataset
    print("\n📥 STAGE 1: Loading MentalChat16K Dataset")
    df = pipeline.load_mentalchat16k_dataset()
    if df is None:
        print("❌ Failed to load dataset. Please check your internet connection.")
        return None, None

    # Stage 2: Dataset analysis
    print("\n🔍 STAGE 2: Comprehensive Dataset Analysis")
    analysis_stats = pipeline.analyze_dataset_characteristics()

    # Stage 3: Create classification tasks
    print("\n🎯 STAGE 3: Creating Classification Tasks")
    classification_metrics = pipeline.create_classification_task()

    # Stage 4: Feature extraction
    print("\n⚙️ STAGE 4: Advanced Feature Extraction")
    feature_cols = pipeline.extract_advanced_features()

    # Stage 5: Train BERT classifier
    print("\n🤖 STAGE 5: Training BERT Classifier")
    bert_model, bert_metrics = pipeline.train_bert_classifier(
        target_column='sentiment_category',
        epochs=3,
        batch_size=16
    )

    # Stage 6: Train feature-based model
    print("\n🧠 STAGE 6: Training Advanced Neural Network")
    feature_model, feature_metrics = pipeline.train_feature_based_model(
        target_column='response_type'
    )

    # Stage 7: Cross-validation analysis
    print("\n📊 STAGE 7: Cross-Validation Analysis")
    cv_metrics = pipeline.cross_validation_analysis(
        target_column='sentiment_category',
        cv_folds=5
    )

    # Stage 8: Generate final report
    print("\n📝 STAGE 8: Generating Research Report")
    final_report = pipeline.generate_research_report()
  
    return pipeline, final_report

# Execute the complete research pipeline
pipeline, results = run_complete_mentalchat16k_research()
