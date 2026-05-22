import pickle
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights
from sklearn.linear_model import LogisticRegression
from app.config import MODEL_PATH, DEVICE
from app.ml.utils import preprocess_image_path
import os

class FeatureExtractor:
    def __init__(self):
        self.device = DEVICE
        # Retrieve pre-trained MobileNetV3 (Small)
        try:
            self.model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            # Fallback for older torchvision versions
            self.model = models.mobilenet_v3_small(pretrained=True)
            
        # Replace classifier head with Identity to extract raw pooled feature maps
        self.model.classifier = torch.nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Extract features for a preprocessed image tensor.
        """
        with torch.no_grad():
            tensor = tensor.to(self.device)
            features = self.model(tensor)
            # Flatten to 1D if necessary, keep batch size dimension
            features = torch.flatten(features, 1)
            return features.cpu().numpy()

class TeachableMachineEngine:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model_data = None
        self.load_model()

    def load_model(self):
        """
        Loads the trained Scikit-Learn classifier and class labels if they exist.
        """
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model_data = pickle.load(f)
            except Exception as e:
                print(f"Failed to load existing model: {e}")
                self.model_data = None
        else:
            self.model_data = None

    def train(self, dataset_dir: str) -> dict:
        """
        Reads images from class folders, extracts features, trains a Logistic Regression
        model, and serializes the result to disk.
        """
        if not os.path.exists(dataset_dir):
            return {"status": "error", "message": "Dataset directory does not exist. Please upload samples first."}

        # Gather class folders
        class_folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        # Validations
        if len(class_folders) < 2:
            return {
                "status": "error", 
                "message": f"To train a classification model, you must define at least 2 distinct classes. Currently found: {len(class_folders)} class(es)."
            }

        X = []
        y = []
        class_names = sorted(class_folders)

        for label_idx, class_name in enumerate(class_names):
            class_path = os.path.join(dataset_dir, class_name)
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            if len(image_files) == 0:
                return {
                    "status": "error",
                    "message": f"Class folder '{class_name}' is empty. Every category must contain at least 1 image sample."
                }
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    # Preprocess and extract features
                    tensor = preprocess_image_path(img_path).unsqueeze(0)
                    features = self.feature_extractor.extract(tensor)
                    X.append(features[0])
                    y.append(label_idx)
                except Exception as e:
                    print(f"Skipping corrupt image {img_file} in class {class_name}: {e}")

        if len(X) == 0:
            return {"status": "error", "message": "No valid training images could be processed."}

        X = np.array(X)
        y = np.array(y)

        # Train Scikit-Learn Logistic Regression Classifier
        # liblinear solver is perfect for small/medium datasets, L2 penalized
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='liblinear')
        clf.fit(X, y)

        # Serialize trained classifier and classes list
        self.model_data = {
            "classifier": clf,
            "classes": class_names
        }

        # Save to disk
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model_data, f)

        return {
            "status": "success",
            "message": f"Model successfully trained on {len(X)} samples across {len(class_names)} classes!",
            "classes": class_names,
            "sample_count": len(X)
        }

    def predict(self, tensor: torch.Tensor) -> dict:
        """
        Performs inference on a preprocessed image tensor and returns class probability mappings.
        """
        if not self.model_data:
            return {"status": "error", "message": "Model weights are not loaded. Please train the model first."}

        # Extract feature vector
        features = self.feature_extractor.extract(tensor)

        # Retrieve trained model and class labels
        clf = self.model_data["classifier"]
        classes = self.model_data["classes"]

        # Run prediction
        probs = clf.predict_proba(features)[0]  # Shape: (num_classes,)
        pred_idx = clf.predict(features)[0]

        predictions = {classes[i]: float(probs[i]) for i in range(len(classes))}
        
        return {
            "status": "success",
            "predicted_class": classes[pred_idx],
            "confidence": float(probs[pred_idx]),
            "predictions": predictions
        }
