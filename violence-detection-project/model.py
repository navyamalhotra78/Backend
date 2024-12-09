import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

class Model:
    def __init__(self, settings_path: str = './settings.yaml'):
        with open(settings_path, "r") as file:
            self.settings = yaml.safe_load(file)

        self.device = self.settings['model-settings']['device']
        self.model_name = self.settings['model-settings']['model-name']
        self.threshold = self.settings['model-settings']['prediction-threshold']
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.labels = self.settings['label-settings']['labels']
        self.labels_ = []
        for label in self.labels:
            text = 'a photo of ' + label  
            self.labels_.append(text)

        self.text_features = self.vectorize_text(self.labels_)
        self.default_label = self.settings['label-settings']['default-label']

    @torch.no_grad()
    def transform_image(self, image_batch: list):
        # Convert each frame from numpy to PIL, preprocess, and stack into batch
        tf_images = [self.preprocess(Image.fromarray(image).convert('RGB')).unsqueeze(0).to(self.device)for image in image_batch]
        return torch.cat(tf_images)  # Stack all tensors along batch dimension

    @torch.no_grad()
    def tokenize(self, text: list):
        text = clip.tokenize(text).to(self.device)
        return text

    @torch.no_grad()
    def vectorize_text(self, text: list):
        tokens = self.tokenize(text=text)
        text_features = self.model.encode_text(tokens)
        return text_features

    @torch.no_grad()
    def predict_(self, text_features: torch.Tensor,image_features: torch.Tensor):
        # Pick the top most similar label for each image in the batch
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        values, indices = similarity.topk(1, dim=-1)
        return values, indices

    @torch.no_grad()
    def predict_batch(self, image_batch: list) -> list:
        # Transform the batch of images
        tf_images = self.transform_image(image_batch)
        
        # Get image features for the batch
        image_features = self.model.encode_image(tf_images)

        # Predict labels for the batch
        predictions = []
        for image_feature in image_features:
            values, indices = self.predict_(text_features=self.text_features, image_features=image_feature.unsqueeze(0))
            label_index = indices[0].cpu().item()
            label_text = self.default_label
            model_confidence = abs(values[0].cpu().item())

            if model_confidence >= self.threshold:
                label_text = self.labels[label_index]

            predictions.append({
                'label': label_text,
                'confidence': model_confidence
            })

        return predictions

    @torch.no_grad()
    def predict(self, image: np.array) -> dict:
        # Transform the image to tensor
        tf_image = self.transform_image([image])  # Transform single image into a batch of size 1

        # Get image features
        image_features = self.model.encode_image(tf_image)

        # Pick the most similar label
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        values, indices = similarity.topk(1, dim=-1)

        # Extract label and confidence
        label_index = indices[0].cpu().item()
        label_text = self.default_label
        model_confidence = abs(values[0].cpu().item())

        if model_confidence >= self.threshold:
            label_text = self.labels[label_index]

        return {
            'label': label_text,
            'confidence': model_confidence
        }

    @staticmethod
    def plot_image(image: np.array, title_text: str):
        plt.figure(figsize=[13, 13])
        plt.title(title_text)
        plt.axis('off')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        
# Code for video capture and batch processing
def process_video_batch(model, video_source=0, batch_size=4, output_path=None):
    cap = cv2.VideoCapture(video_source)  
    frame_batch = []
    frame_index = 0

    # Prepare output video writer if needed
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        # Convert the frame to RGB if using OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_batch.append(frame)

        if len(frame_batch) == batch_size:
            # Process the batch of frames
            predictions = model.predict_batch(frame_batch)
            
            for i, prediction in enumerate(predictions):
                frame_index += 1
                print(f"Frame {frame_index}: {prediction}")

                annotated_frame = cv2.putText(cv2.cvtColor(frame_batch[i], cv2.COLOR_RGB2BGR),
                f"{prediction['label']} ({prediction['confidence']:.2f})",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,cv2.LINE_AA)

                if out:
                    out.write(annotated_frame)

                cv2.imshow("Video", annotated_frame)

            frame_batch = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = Model()

    # video_file = "./data/office_fight.mp4"  
    video_file = 0 #webcam
    process_video_batch(model, video_source=video_file, batch_size=4, output_path=None)