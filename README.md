## PDF Chat with AWS Bedrock 💬
This project allows you to chat with PDF files using AWS Bedrock services and models such as Claude v2 and Llama2. The system reads and splits PDFs into chunks, stores them as embeddings using FAISS, and retrieves relevant chunks to generate answers using AWS-provided large language models (LLMs).

Table of Contents
Prerequisites
Installation
AWS Setup
Step 1: Create an IAM User
Step 2: Configure AWS CLI
How to Run
Usage
Key Concepts
Prerequisites
Before setting up this project, make sure you have the following tools installed:

Python 3.8 or higher
AWS CLI (Command Line Interface)
Streamlit (for the web app interface)
Boto3 (AWS SDK for Python)
Installation
To get started, clone this repository and install the necessary Python dependencies.

bash
Copy code
# Clone the repository
git clone https://github.com/yourusername/pdf-chat-bedrock.git
cd pdf-chat-bedrock

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required Python packages
pip install -r requirements.txt

AWS Setup

Step 1: Create an IAM User
You will need to create an IAM (Identity and Access Management) user in AWS with access to Amazon Bedrock and S3.

Sign in to the AWS Management Console and navigate to the IAM service.
Create a new user:
Go to Users > Add user.
Enter a username (e.g., bedrock-user).
Select Programmatic access.
Set permissions:
Attach the following policies to the user:
AmazonS3FullAccess (for storing PDF data if needed)
AmazonBedrockFullAccess (or custom policies for Bedrock if applicable)
Save the Access Keys:
After creating the user, download the .csv file containing the Access Key ID and Secret Access Key.
These keys are required to authenticate and communicate with AWS services programmatically.

Step 2: Configure AWS CLI
After creating the IAM user, you need to configure the AWS CLI with your credentials.

Install AWS CLI:

Follow the instructions to install the AWS CLI from here.
Configure AWS CLI using the access keys you generated:

bash
Copy code
aws configure
You will be prompted to enter the following information:

AWS Access Key ID
AWS Secret Access Key
Default region name (e.g., us-east-1)
Default output format (e.g., json)
Verify Configuration: You can check if the AWS CLI is configured properly by running:

bash
Copy code
aws sts get-caller-identity
This should return information about your IAM user, such as user ID and account number.

How to Run
Once AWS is configured and the project dependencies are installed, follow these steps to run the application:

Start the Streamlit app:

bash
Copy code
streamlit run app.py
The app will open in your browser at http://localhost:8501.

Load PDFs into the data folder in the project root directory. You can add multiple PDF files, and they will be ingested, split, and embedded into vectors.

Update or Create Vector Store:

On the sidebar, click "Vectors Update" to create the vector store for the PDFs.
This will process the documents, generate embeddings, and store them locally in FAISS.
Ask a Question:

Enter a question related to the PDFs in the input box.
You can choose to get responses from either Claude v2 or Llama2 by clicking the corresponding buttons.
Usage
Claude v2 Output: This uses the Claude v2 LLM from Anthropic, available through AWS Bedrock.
Llama2 Output: This uses Meta's Llama2 model to generate responses based on your question.
When you click any of these buttons, the app retrieves relevant chunks from the PDFs, passes them as context to the LLM, and provides a response.

## Key Concepts
Bedrock: A managed service by AWS to access foundational models like Claude and Llama2.
Titan Embeddings: A model used to generate text embeddings for documents that can later be used to perform semantic search and retrieval.
FAISS (Facebook AI Similarity Search): A library for efficient similarity search. It's used to store document embeddings and retrieve relevant chunks.
Streamlit: A Python-based framework for building interactive web apps with minimal code.
Example of AWS Configuration
Here’s an example configuration when using aws configure:

bash
Copy code
$ aws configure
AWS Access Key ID [None]: AKIAXXXXXXXXEXAMPLE
AWS Secret Access Key [None]: abcdefghijklmnopqrstuvwxexample
Default region name [None]: us-east-1
Default output format [None]: json
Make sure to save your access keys securely, and never hardcode them into your project.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

If you run into any issues, feel free to submit an issue in the GitHub repository or contact the project maintainers.

# Enjoy using Chat with PDF using AWS Bedrock! 🎉
<img width="959" alt="image" src="https://github.com/user-attachments/assets/955de454-1286-4458-a45b-9c37e67463ee">
