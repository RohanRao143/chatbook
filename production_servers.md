Specialized ML platforms
These platforms offer specific tools and hardware tiers for deploying machine learning models, and they are generally the best choice for showcasing your work for free. 
1. Hugging Face Spaces
Best for: Demos, prototypes, and sharing with the ML community.
Key features:
Allows deploying interactive apps using a custom Dockerfile.
Provides a "CPU Basic" free hardware tier with a generous 16 GB RAM, which can accommodate your BART and Faiss models.
Integrates with the Hugging Face Hub, so you can easily pull your models without them needing to be part of the final image.
Limitations: Geared toward public, non-commercial use on the free plan. 
2. Streamlit Community Cloud 
Best for: Creating and deploying interactive web apps from a GitHub repository.
Key features:
Streamlit simplifies the process of turning your Python code into a shareable web app.
It automatically handles the build process, which can simplify deployment.
Limitations: Free apps can go to sleep after an hour of inactivity. It also has memory limitations that may be exceeded by your model and dependencies. 
General-purpose cloud with generous free tiers
If you want a traditional cloud server, these providers offer more resources in their free tiers than Railway, but you'll need to use advanced deployment techniques.
3. Oracle Cloud Infrastructure (OCI)
Best for: A traditional VM environment with the most generous "always free" tier resources.
Key features:
The "Always Free" tier offers a 4-core ARM VM with 24 GB RAM, which is ample for your large models.
Includes 200 GB of total storage, which can be allocated to persistent block volumes.
Considerations:
You must manually provision and manage the server, which is more complex than a platform-as-a-service (PaaS) like Railway.
You will still need to optimize your Docker image size and use a persistent volume to store your models to comply with the storage limits. 
4. Google Cloud Platform (GCP)
Best for: Modern, serverless deployments for event-driven or lightweight APIs.
Key features:
The Free Tier provides $300 in credit for new customers, which can be used on more powerful virtual machines.
The "Always Free" tier includes Cloud Run, a serverless container platform that handles up to 2 million requests per month for free, but it has a memory limit of 512 MB and may not be suitable for your models.
Considerations: Your large model and its memory requirements will likely exceed the limits of the "Always Free" tier. The initial free credit is the only way to get a larger machine without payment. 
A deployment strategy for free services 
Regardless of the platform you choose, you will need to fundamentally change your deployment strategy to accommodate your large models on a free service.
Rebuild a smaller Docker image: Create an extremely minimal image for your application code. Exclude the model weights and data from the build process.
Use a cloud storage solution: Store your 13.5 GB BART model on a cloud storage service like OCI Object Storage (free tier) or GCP Cloud Storage.
Mount a persistent volume (for Oracle Cloud): If you use an OCI VM, you can attach an "Always Free" block volume to your server. During deployment, download the model from cloud storage to this volume.
Download models at runtime: For containerized solutions like Hugging Face Spaces or GCP Cloud Run, write a script to download the model from cloud storage to the container's disk on startup. This is a common pattern for handling large assets.
This process separates your large assets from your deployable application, enabling you to use the limited free resources offered by these platforms. 
