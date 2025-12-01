### General
A proof-of-concept app in Python (using Streamlit for UI) for generating short journalistic captions and tags for images. The current verison is based on Google Cloud Vision API + OpenAI API. Azure Image Analysis was also extensively tested; an alternative Azure implementation exists on a branch. Sorry, for now the UI and the output captions/tags are **Polish only**! English language support will be implemented soon.  

Demo currently deployed to Streamlit Community Cloud: [check it out](https://photo-assistant-yjmeumynuaydjxhcu43swk.streamlit.app). This one will not store your images/data.


### Components/files
- **app.py** - the main streamlit app
- **config.py** - reads secrets either from .env or from streamlit secrets
- **content_safety_google.py** - defines ContentSafetyChecker class for obtaining Google SafeSearch evaluation results of an image. Used for content alerts in UI & providing safety context for LLM.
- **faq_system.py** - defines FAQSystem class for storage of FAQs and their embeddings, semantic search and answer generation using LLM
- **image_analyzer.py** - ImageAnalyzer class using Google Cloud Vision. Handles labels, faces, objects, landmarks & OCR, incorporates them in LLM prompt along with user context and safety context.
- **web_entity_detector.py** - defines WebEntityDetector class, which uses Google Cloud Vision Web Detection API to obtain 'best guess label' and the most probable related entity descriptions. They are combined and provided to the user in UI as suggested context. Once approved/edited, they will become 'user context' in the LLM prompt.
- **faq_data.json** - current set of FAQs. When this is modified, FAQSystem will generate new faq_embeddings_cache.pkl


### How to run
To run the app locally, you need to have an OpenAI API key and a Google Cloud project with Cloud Vision API enabled.  
Install `uv`, clone the repo and:  
```
uv sync
uv run --env-file .env streamlit run app.py
```

The root folder of the repo must contain your `google-credentials.json` and your `.env` file with your secrets:
```
GOOGLE_APPLICATION_CREDENTIALS=google-credentials.json
GOOGLE_PROJECT_ID=
OPENAI_API_KEY=
```

### Also, feel free to play around online with my earlier test deployments (please note these WILL store your uploads and outputs for review/debugging for some time):  
- [Azure-based test deployment](https://photo-assistant-mtnmekjxqywnved9t7et8p.streamlit.app) leveraging the cool Dense Captions feature of Azure Image Analysis API 4.0  
- [GCP-based test deployment](https://photo-assistant-ggooggllee-ved9t7et8p.streamlit.app) of the current app showcasing the power (and threats) of Web-Detection-obtained context (unlike to the current demo deployment, this one DOES store uploads for future review)


### Azure vs. Google comparative analysis (or some fun examples)

**1) The Hitler case**




