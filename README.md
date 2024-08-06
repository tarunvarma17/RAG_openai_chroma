# RAG_openai_chroma
Answer questions from pdf using open ai embeddings, gpt3.5 turbo, and chromadb vectorstore


Created a Linux VM on azure. 
Had to choose the zone as central india, as none of the vm's were available in any of the other zones
Selected the zone 1 (default)
The vm that we opted for was d4s v3
This has 4vcpus, and 16GB memory
There are 2 options - ssh key pair, or password. As I was more comfortable with password, I chose that
Created username, pwd, and got a public IP.
Ssh'ed in using these three as I did for opsassistant
 
(NOTE: I did this initially with a smaller VM (1 cpu, 1gb) and that didn’t work because vs code automatically downloads things more than that)
 
Only thing is, I believe public IP address will keep changing, and we have to pay more for that to remain constant
Before I could pip install all the required docs, I had to do the following:
sudo apt-get update.
Then, I had to install pip using sudo apt-get python3-pip
I am able to use the libraries I downloaded, but vscode is still showing them as yellow underline
 
To get all the libraries I actually need, I copied the requirements.txt file
I made a virtual environment with python with python3 -m venv .venv
Then, I selected that venv from the bottom, and closed the terminal and opened a new one
Now, with the requirements.txt file, I couldn’t simply do pip install -r requirements.txt
Instead I had to do pip freeze > requirements.txt
And THEN pip install -r requirements.txt
 
Coming back, this did not seem to work
I manually installed them all
The virtual environment was setup wrong. Did it again using the python environment manager plugin
First step is getting the raw text. We do this by first
Importing pypdf
Opening the file using with open(filepath, 'rb') as file: (file has to be opened in read binary form)
 
You can also use langchain_community.documnet_loaders to get data from all different forms
(includes PyPDF loader)
 
Then create an instance of the reader as reader = pypdf2.pdfreader(file)
Text = ''
For page in reader.pages:
Text+=page.extract_text()
 
Here, the extract_text returns the text and the page does this page by page
 
Then,
We have to perform chunking. There are many ways to do this. 
First is by using sentence tokenize, which just splits them by sentences and puts n sentences into one chunk
So, in this case if you say chunk_size = 5, it puts 5 sentences in one chunk
 
A better way is using langchains recursive character text splitter. Here you specify number of tokens and overlap
So, if chunk sieze = 1000 and overlap = 100, each chunk will be of size 1000 tokens, and the last 100 will overlap with first 100 of next chunk
 
I then did this another way using the langchain_community document loaders way. This creates documents for each page (page content, page_number etc). Very convenient. And instead of splittting from text, we split from documents
My next step was creating the embeddings. So the first thing to check is if the embeddings for a certain file are created, then we don’t need to take costly computational power to compute them again. So, instead we use a pickle file to store the vector stores of these files
 
I'll get back to what vector stores are
To perform embeddings, I used sentencetransformer
Basically all I have to do is define the model and then model.encode(chunks)
 
Now, a vectorstore is a vector db where we store all the embeddings which allows for quick lookup when a query comes in. Its essentially just a convenient data structure that holds our embeddings. This process of making embeddings (or any data) easy to match with a query is called indexing
 
If the filename is test.pdf, we check if test.pkl is there in the embeddings folder (test.pkl should hold the vectorstore for that file) using os.path.exists
 
If it does, then we pickle.load the file into vectorspace variable
Else, we perform embeddings, get vectorstore , and write the file using pickle.dump
We get vectorstore using 
 
vectorstore = faiss.IndexFlatL2(embeddings.shape[1])
vectorstore.add(embeddings)
First we have to perform embeddings on the query
It is essential to make sure this is of the right shape
Since there is just one query, its shape will be (384,)
However, it should be (1, 384) or rather (#queries, embedding size)
 
Then, we get the top chunks using vectorstore.search(query_embeddings, k)
Where k is how many chunks we want to return
This returns the distances and indices of the chunks  (lower distance, better answer)
 
Now to get the answer, we formulate all these into a context and query pair
I used bert transformer llm
Initialise it by saying nlp = pipeline("question-answering")
Result = nlp(question = query, context =  context)
To run streamlit on vm, we cannot simply use streamlit run app.py
 
Instead we have to specify
python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
 
We set it up like this because azure vm uses only port 8000 and 443 by default
Now, the way I wanted to do this was to save the pdf in the pdf files folder
I did this by creating a function that opened the filepath and then wrote into this using .getbuffer()
 
So, if our folder path is …/pdf_files/
And uploaded_file.name = test.pdf
Then we do with open(…./pdf_files/test.pdf, "wb") as f:
f.write(uploaded_file.getbuffer())
Return filepath
 
Now , we can get the pdf using
Pdf = st.file_uploader("text to display", "pdf")
And now while this is there
If pdf is not None:
….do everything…..
For this stage, I kept everything the same that is, the embeddings, chunking etc
I only wanted to change it so that the BERT llm is replaced with open ai's llm
The first problem I face, was resolved by simply changing the version of my langchain and openai libraries
The first change I had to make was to make sure my 'context' was a list of strings rather than just one long string
After that, this is the most important step
I had to change it from a string type to a document type. To accomplish this, I had to download the Documents library from langchain. And then convert each doc (each item in the list of strings) to a document form
 
 
documents = [Document(page_content=doc, metadata={"source": "local"}) for doc in docs]
I don’t have to fill meta data, but I did anyways
 
Next, I inititalised the model as llm = OpenAI("model_name" = 'gpt-3.5-turbo')
Next I had to load the qa_chain and specify the llm and chain_type
Next, response = chain.run(input_documents = context, question = query)
 
Additionally, as OpenAI stopped giving out free credits, I had to purchase $5 worth credits to run this
The model im using costs $3 for every 1 million tokens processed. My input is about 400-600 tokens each time
My first aim is to stick with FAISS (note, this is FAISS not faiss, as I am using langchain)
And I want to compare the different sizes of the vectorstores with each step
 
First thing to know about the difference is that chroma by default works in ephemeral mode
This means it doesn’t store the data on disk, unlike what we were doing with FAISS. We need to change it to persist mode to tackle that.
 
One difference is, when we use persist mode these vectorstores get added to memory directly. So, we don’t have to use pkl files like  do in faiss. When we create a new vectore store, we pass the chunks, embedding function, and the persist directory. It automatically does the embedding from this step
 
You can pass the dir as ./Embeddings/{filename}. Now when this is created, a folder under Embeddings will exist for that filename. If this exists, we can do vectorstores = Chroma(persist_dir = …., embedding = ef)
 
Its slightly confusing that we have to pass in the embeddings again, but it doesn’t compute them again when we call it from disk, but rather just checks if the embeddings returned are of the correct type
To have more control on our output, we can use chains to get the llm response
 
First, we create a document chain, which takes in the llm and the prompt
Prompt is created using .from_template()
Here, jinja formatting is used for the context and for the query, rest is hardcoded in
 
Once we build this, we get our context by defining a retriever. A retriever basically does all the matching of the top responses given a vectorstore and the document_chain
 
Retriever = index.as_retriever()
Retrieval_chian = create_retrieval_chain(retriever, document_chain)
 
Now, the retrieval chain has everything - the index (from retriver), and the llm and prompt
The prompt can then be fitted with the query using invoke
 
Response = retrieval_chain.invoke({"input":query})


