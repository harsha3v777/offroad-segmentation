
Live Demo - https://huggingface.co/spaces/harsha3777/offroad-segmentation

🎬 OffRoad Vision 

🎙️ INTRO 

"Hello everyone! Today I'm going to walk you through our project — OffRoad Vision, an AI-powered terrain segmentation system that we built for the Duality AI Hackathon.
The goal of this project is simple — given any off-road scene image, our model automatically identifies and labels every terrain type visible in the image, pixel by pixel.
Let's dive in."


🗂️ PROBLEM STATEMENT

"Off-road environments are highly unstructured. Unlike city roads, they contain a mix of trees, rocks, dry grass, bushes, logs, and sky — all blended together.
For autonomous vehicles or robots operating in such environments, understanding the terrain is critical. They need to know — what is ground? What is an obstacle? What is safe to drive on?
This is exactly the problem semantic segmentation solves — classifying every single pixel in the image into a terrain category."


📦 DATASET 

"The dataset was provided by Duality AI as part of the hackathon.
It contains:

2,857 training images with ground truth segmentation masks
317 validation images
1,002 test images

Each image is an off-road scene captured from a vehicle camera at 960 by 540 resolution.
The dataset has 11 terrain classes — Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, and Sky.
One important fix we made — the original script was missing class 600 which is Flowers. We caught this bug and added it, bringing the total from 10 to 11 classes."


🧠 MODEL ARCHITECTURE

"Our model has two parts.
Part 1 — The Backbone: DINOv2
We used Facebook's DINOv2 Vision Transformer, specifically the small variant — vits14. This is a powerful self-supervised model pretrained on millions of images. It acts as a feature extractor — it takes the image and produces rich patch-level embeddings.
We kept the backbone frozen during training — meaning we did not update its weights. This saved a lot of computation.
Part 2 — The Segmentation Head: ConvNeXt-style
On top of DINOv2, we added a lightweight ConvNeXt-style segmentation head. This head takes the patch embeddings from DINOv2, reshapes them into a 2D feature map, and applies depthwise convolutions to produce per-pixel class predictions for all 11 classes.
The total trainable parameters were only about 2.4 million — very lightweight on top of a powerful backbone."


⚙️ TRAINING PROCEDURE 

"We trained the model on Google Colab using a Tesla T4 GPU.
Key training details:

25 epochs of training
Batch size of 2
AdamW optimizer with learning rate 1e-4 and cosine annealing scheduler
Weighted Cross Entropy Loss — we gave higher weights to rare classes like Flowers and Logs so the model doesn't ignore them
Images were resized to 476 by 266 pixels — divisible by 14 for DINOv2's patch size

We also implemented best model checkpointing — saving the model whenever validation IoU improved.
Training took approximately 50 minutes per epoch, so the full training run was around 20 hours."


📊 RESULTS 

"Here are our results:

Best Validation mIoU: 0.3125 — achieved at epoch 21
Validation Pixel Accuracy: 67.19%

On the 3 test images we evaluated:

Image 1: IoU 0.2277
Image 2: IoU 0.2594
Image 3: IoU 0.1872
Average test IoU: 0.2248

The model successfully identifies dominant classes like Trees, Sky, and Landscape. Rare classes like Flowers and Logs are harder due to limited training examples.
As you can see in these predictions — the colored overlay correctly segments the sky in blue, trees in green, rocks in grey, and ground in tan."


🌐 DEPLOYMENT

"We didn't stop at just training the model. We built and deployed a full web application.
The backend is a Flask server that loads the DINOv2 backbone and our segmentation head, accepts an uploaded image, runs inference, and returns the result.
The frontend is a clean HTML website where you can drag and drop any off-road image and instantly see:

The original image
The segmentation mask with color-coded terrain classes
An overlay of mask on original image
A pixel distribution chart showing what percentage of the image is each terrain type

We deployed this live on Hugging Face Spaces using Docker, so anyone in the world can access it right now at this URL.
The code is also open source on GitHub."


🔮 FUTURE WORK

"While we're proud of what we built, there's a lot of room to improve. Here's what we plan to do next:
1. Larger Backbone
We used DINOv2 small. Upgrading to DINOv2 base or large would significantly improve feature quality and IoU scores.
2. Better Segmentation Head
We could replace our simple ConvNeXt head with a full decoder like SegFormer or DeepLabV3+ for sharper boundary predictions.
3. Data Augmentation
Adding augmentations like random flips, color jitter, and random crops during training would help the model generalize better to unseen terrain.
4. Fine-tuning the Backbone
Currently the backbone is frozen. Unfreezing and fine-tuning it on our specific terrain data with a very small learning rate could push IoU significantly higher.
5. Real-time Video Segmentation
Extending the web app to process live video streams or drone footage would make this useful for real autonomous vehicle applications.
6. Generate All Test Predictions
We only evaluated 3 test images. The next step is running inference on all 1,002 test images and submitting full predictions to the hackathon leaderboard."


🎬 OUTRO

"To summarize — we built an end-to-end terrain segmentation pipeline using DINOv2 and a custom ConvNeXt head, trained it on the Duality AI dataset, and deployed it as a live web application.
Thank you for watching. The live demo is available at our Hugging Face Space, and all code is on GitHub. Feel free to try it out!"
