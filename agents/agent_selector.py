from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.data_loader import DataLoader


class AgentSelector:
    def __init__(self, user_input):
      self.parameters = user_input['parameters']
      self.data_path_train = user_input['dataset_train']
      self.data_path_test = user_input['dataset_test']

      self.load_data()

      # self.package_name = "pygod" if user_input['dataset_train'].endswith(".pt") else "pyod"
      self.package_name = "pygod" if type(self.y_train) is str and self.y_train == 'graph' else "pyod"
      # print(f"Package name: {self.package_name}")

      self.tools = self.generate_tools(user_input['algorithm'])
      self.documents = self.load_and_split_documents()
      self.vectorstore = self.build_vectorstore(self.documents)

    def load_data(self):
      train_loader = DataLoader(user_input['dataset_train'], store_script=True, store_path='train_data_loader.py')
      test_loader = DataLoader(user_input['dataset_test'], store_script=True, store_path='test_data_loader.py')

      X_train, y_train = train_loader.load_data(split_data=False)
      X_test, y_test = test_loader.load_data(split_data=False)
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test

    def load_and_split_documents(self,folder_path="./docs"):
      """
      load ./docs txt doc, divided into small blocks。
      """
      documents = []
      text_splitter = CharacterTextSplitter(separator="\n", chunk_size=700, chunk_overlap=150)

      for filename in os.listdir(folder_path):
         if filename.startswith(self.package_name):
               file_path = os.path.join(folder_path, filename)
               with open(file_path, "r", encoding="utf-8") as file:
                  text = file.read()
                  chunks = text_splitter.split_text(text)
                  documents.extend(chunks)

      return documents
    def build_vectorstore(self,documents):
      """
      The segmented document blocks are converted into vectors and stored in the FAISS vector database.
      """
      embedding = OpenAIEmbeddings()
      vectorstore = FAISS.from_texts(documents, embedding)
      return vectorstore
    def generate_tools(self,algorithm_input):
      """Generates the tools for the agent."""
      if algorithm_input[0].lower() == "all":
        if self.package_name == "pygod":
          return ['SCAN','GAE','Radar','ANOMALOUS','ONE','DOMINANT','DONE','AdONE','AnomalyDAE','GAAN','DMGD','OCGNN','CoLA','GUIDE','CONAD','GADNR','CARD']
        else:
          return ['ECOD', 'ABOD', 'FastABOD', 'COPOD', 'MAD', 'SOS', 'QMCD', 'KDE', 'Sampling', 'GMM', 'PCA', 'KPCA', 'MCD', 'CD', 'OCSVM', 'LMDD', 'LOF', 'COF', '(Incremental) COF', 'CBLOF', 'LOCI', 'HBOS', 'kNN', 'AvgKNN', 'MedKNN', 'SOD', 'ROD', 'IForest', 'INNE', 'DIF', 'FeatureBagging', 'LSCP', 'XGBOD', 'LODA', 'SUOD', 'AutoEncoder', 'VAE', 'Beta-VAE', 'SO_GAAL', 'MO_GAAL', 'DeepSVDD', 'AnoGAN', 'ALAD', 'AE1SVM', 'DevNet', 'R-Graph', 'LUNAR']
      return algorithm_input

if __name__ == "__main__":
  import sys
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  from config.config import Config
  os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY

  user_input = {
    "algorithm": ["ALL"],
    "dataset_train": "./data/inj_cora_train.pt",
    "dataset_test": "./data/inj_cora_test.pt",
    "parameters": {
      "contamination": 0.1
    }
  }
  agentSelector = AgentSelector(user_input= user_input)
  print(f"Tools: {agentSelector.tools}")