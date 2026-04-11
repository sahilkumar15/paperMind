# cd scholarmind
# pip install -r requirements.txt

# cp .env.example .env
# # Edit .env → add GROQ_API_KEY=gsk_your_key_here

# # Pre-build KatzBot index (saves to disk, loads instantly after)
rmdir /s /q katzbot\faiss_index
del katzbot\pages_cache.json
python katzbot/build_index.py --refresh --test

# # Run
# streamlit run app.py