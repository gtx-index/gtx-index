import requests
from bs4 import BeautifulSoup
import os
from flask import Flask, jsonify
from flask_cors import CORS
import datetime
from collections import deque
import re
import logging
import tweepy
import resend

# ---------- AI PROVIDERS ----------
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY_2")

groq_client = None
if GROQ_API_KEY and GROQ_API_KEY.strip():
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logging.info("Groq client initialized with primary key")
    except Exception as e:
        logging.warning(f"Failed to initialize Groq with primary key: {e}")
elif GROQ_API_KEY_2 and GROQ_API_KEY_2.strip():
    try:
        groq_client = Groq(api_key=GROQ_API_KEY_2)
        logging.info("Groq client initialized with secondary key")
    except Exception as e:
        logging.warning(f"Failed to initialize Groq with secondary key: {e}")
else:
    logging.warning("No valid GROQ_API_KEY found; Groq will be skipped.")

from google import genai
GEMINI_KEY_1 = os.environ.get("GEMINI_API_KEY_1")
GEMINI_KEY_2 = os.environ.get("GEMINI_API_KEY_2")
GEMINI_KEY_3 = os.environ.get("GEMINI_API_KEY_3")

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

score_history = deque(maxlen=7)

# ---------- HELPERS ----------
def fetch_soup(url, timeout=15):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    r = requests.get(url, timeout=timeout, headers=headers)
    return BeautifulSoup(r.text, 'html.parser')

def extract_text(soup, max_chars=4000):
    # Try common content containers
    for selector in ['article', 'div#content', 'div.content', 'main', 'div.article-body', 'div.story-body']:
        tag = soup.select_one(selector)
        if tag:
            text = tag.get_text(separator=' ', strip=True)
            if len(text) > 100:  # Only return if it has substantial text
                return text[:max_chars]
    
    # Fallback: get all paragraph text
    paragraphs = soup.find_all('p')
    if paragraphs:
        text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        if len(text) > 100:
            return text[:max_chars]
    
    # Ultimate fallback: just get the body text
    body = soup.find('body')
    if body:
        return body.get_text(separator=' ', strip=True)[:max_chars]
    
    return ""

def looks_like_individual_doc(url):
    if any(kw in url for kw in ['rss', '.xml']):
        return False
    return True

# ---------- SOURCE SCRAPERS (GEOPOLITICAL) ----------
def scrape_state_dept():
    """Scrape US State Department press briefings."""
    sources = []
    try:
        soup = fetch_soup("https://www.state.gov/briefings/")
        items = soup.select('a[href*="/briefings/"]')
        for a in items[:2]:
            href = a.get('href')
            if href:
                full_url = href if href.startswith('http') else "https://www.state.gov" + href
                title = a.get_text(strip=True)
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'state_dept', 'title': title, 'url': full_url})
        logging.info(f"Scraped {len(sources)} State Dept briefings")
    except Exception as e:
        logging.error(f"State Dept scrape error: {e}")
    return sources

def scrape_china_mfa():
    """Scrape Chinese Ministry of Foreign Affairs statements."""
    sources = []
    try:
        soup = fetch_soup("https://www.fmprc.gov.cn/eng/xw/")
        items = soup.select('a[href*="/xw/"]')
        for a in items[:2]:
            href = a.get('href')
            if href:
                full_url = "https://www.fmprc.gov.cn" + href if href.startswith('/') else href
                title = a.get_text(strip=True)
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'china_mfa', 'title': title, 'url': full_url})
        logging.info(f"Scraped {len(sources)} China MFA statements")
    except Exception as e:
        logging.error(f"China MFA scrape error: {e}")
    return sources

def scrape_rus_mid():
    """Scrape Russian Ministry of Foreign Affairs statements (English)."""
    sources = []
    try:
        soup = fetch_soup("https://mid.ru/en/press_service/spokesman/")
        items = soup.select('a[href*="/spokesman/"]')
        for a in items[:2]:
            href = a.get('href')
            if href:
                full_url = "https://mid.ru" + href if href.startswith('/') else href
                title = a.get_text(strip=True)
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'rus_mid', 'title': title, 'url': full_url})
        logging.info(f"Scraped {len(sources)} Russian MFA statements")
    except Exception as e:
        logging.error(f"Russian MFA scrape error: {e}")
    return sources

def scrape_eu_eeas():
    """Scrape EU External Action Service statements."""
    sources = []
    try:
        soup = fetch_soup("https://www.eeas.europa.eu/eeas/statements_en")
        items = soup.select('a[href*="/statements/"]')
        for a in items[:2]:
            href = a.get('href')
            if href:
                full_url = "https://www.eeas.europa.eu" + href if href.startswith('/') else href
                title = a.get_text(strip=True)
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'eu_eeas', 'title': title, 'url': full_url})
        logging.info(f"Scraped {len(sources)} EU EEAS statements")
    except Exception as e:
        logging.error(f"EU EEAS scrape error: {e}")
    return sources

def scrape_military_briefings():
    """Scrape Pentagon / military briefings."""
    sources = []
    try:
        soup = fetch_soup("https://www.defense.gov/News/Releases/")
        items = soup.select('a[href*="/News/Releases/"]')
        for a in items[:2]:
            href = a.get('href')
            if href:
                full_url = "https://www.defense.gov" + href if href.startswith('/') else href
                title = a.get_text(strip=True)
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'military_briefing', 'title': title, 'url': full_url})
        logging.info(f"Scraped {len(sources)} military briefings")
    except Exception as e:
        logging.error(f"Military scrape error: {e}")
    return sources

def scrape_state_media():
    """Scrape Xinhua (China) and TASS (Russia) headline statements."""
    sources = []
    try:
        soup = fetch_soup("http://www.xinhuanet.com/english/")
        items = soup.select('a[href*="/english/"]')
        for a in items[:1]:
            href = a.get('href')
            if href:
                full_url = "http://www.xinhuanet.com" + href if href.startswith('/') else href
                title = a.get_text(strip=True)
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'xinhua', 'title': title, 'url': full_url})
    except Exception as e:
        logging.warning(f"Xinhua scrape error: {e}")
    
    try:
        soup = fetch_soup("https://tass.com/politics")
        items = soup.select('a[href*="/politics/"]')
        for a in items[:1]:
            href = a.get('href')
            if href:
                full_url = "https://tass.com" + href if href.startswith('/') else href
                title = a.get_text(strip=True)
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'tass', 'title': title, 'url': full_url})
    except Exception as e:
        logging.warning(f"TASS scrape error: {e}")
    
    logging.info(f"Scraped {len(sources)} state media headlines")
    return sources

# ---------- AI SCORING (GEOPOLITICAL PROMPT) ----------
def score_text_with_ai(text):
    if not text:
        return None
    prompt = f"""
You are a geopolitical risk analyzer. Rate the following text on a scale from 0 (no geopolitical risk, suggesting peace, stability, and cooperation) to 100 (extreme geopolitical risk, suggesting open conflict, war, sanctions, or major escalation). Consider language about military actions, diplomatic tensions, sanctions, territorial disputes, and alliance shifts. Return ONLY the number, no explanation.

Text:
{text[:3000]}
"""
    if groq_client:
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0,
                max_tokens=5
            )
            score_str = chat_completion.choices[0].message.content.strip()
            digits = re.findall(r'\d+', score_str)
            if digits:
                score = int(digits[0])
                logging.info(f"AI score (Groq): {score}")
                return max(0, min(100, score))
        except Exception as e:
            logging.warning(f"Groq failed ({e}), falling back to Gemini-1...")
    if GEMINI_KEY_1:
        try:
            gemini_client = genai.Client(api_key=GEMINI_KEY_1)
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0, "max_output_tokens": 5}
            )
            score_str = response.text.strip()
            digits = re.findall(r'\d+', score_str)
            if digits:
                score = int(digits[0])
                logging.info(f"AI score (Gemini-1): {score}")
                return max(0, min(100, score))
        except Exception as e:
            logging.warning(f"Gemini-1 failed ({e}), falling back to Gemini-2...")
    if GEMINI_KEY_2:
        try:
            gemini_client = genai.Client(api_key=GEMINI_KEY_2)
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0, "max_output_tokens": 5}
            )
            score_str = response.text.strip()
            digits = re.findall(r'\d+', score_str)
            if digits:
                score = int(digits[0])
                logging.info(f"AI score (Gemini-2): {score}")
                return max(0, min(100, score))
        except Exception as e:
            logging.warning(f"Gemini-2 failed ({e}), falling back to Gemini-3...")
    if GEMINI_KEY_3:
        try:
            gemini_client = genai.Client(api_key=GEMINI_KEY_3)
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0, "max_output_tokens": 5}
            )
            score_str = response.text.strip()
            digits = re.findall(r'\d+', score_str)
            if digits:
                score = int(digits[0])
                logging.info(f"AI score (Gemini-3): {score}")
                return max(0, min(100, score))
        except Exception as e:
            logging.error(f"All AI providers failed: {e}")
            return None
    logging.error("No Gemini API keys configured")
    return None

# ---------- MARKET EXPECTATION (VIX + GOLD + DEFENCE ETF) ----------
def compute_market_gtx():
    """Return a 0–100 score derived from VIX, gold price, and defence ETF flows."""
    vix_change = 0
    gold_change = 0
    defence_change = 0

    # 1. VIX change (volatility proxy)
    try:
        vix_url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX?interval=1d&range=1d"
        resp = requests.get(vix_url, timeout=15)
        if resp.status_code == 200:
            data = resp.json().get("chart", {}).get("result", [])
            if data:
                meta = data[0].get("meta", {})
                prev = meta.get("previousClose")
                curr = meta.get("regularMarketPrice")
                if prev and curr:
                    vix_change = ((curr / prev) - 1) * 100
    except Exception as e:
        logging.warning(f"VIX error: {e}")

    # 2. Gold change (safe‑haven proxy)
    try:
        gold_url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1d&range=1d"
        resp = requests.get(gold_url, timeout=15)
        if resp.status_code == 200:
            data = resp.json().get("chart", {}).get("result", [])
            if data:
                meta = data[0].get("meta", {})
                prev = meta.get("previousClose")
                curr = meta.get("regularMarketPrice")
                if prev and curr:
                    gold_change = ((curr / prev) - 1) * 100
    except Exception as e:
        logging.warning(f"Gold error: {e}")

    # 3. Defence ETF (ITA) change – proxy for geopolitical risk pricing
    try:
        ita_url = "https://query1.finance.yahoo.com/v8/finance/chart/ITA?interval=1d&range=1d"
        resp = requests.get(ita_url, timeout=15)
        if resp.status_code == 200:
            data = resp.json().get("chart", {}).get("result", [])
            if data:
                meta = data[0].get("meta", {})
                prev = meta.get("previousClose")
                curr = meta.get("regularMarketPrice")
                if prev and curr:
                    defence_change = ((curr / prev) - 1) * 100
    except Exception as e:
        logging.warning(f"Defence ETF error: {e}")

    # Map to 0–100 scores
    vix_score = min(100, max(0, 50 + vix_change * 50))
    gold_score = min(100, max(0, 50 + gold_change * 100))
    defence_score = min(100, max(0, 50 + defence_change * 100))

    market_gtx = round((vix_score + gold_score + defence_score) / 3, 1)

    if market_gtx <= 20:
        label = "Extremely Low Risk"
    elif market_gtx <= 40:
        label = "Low Risk"
    elif market_gtx <= 60:
        label = "Moderate Risk"
    elif market_gtx <= 80:
        label = "High Risk"
    else:
        label = "Extreme Risk"

    return market_gtx, label, {
        "vix_change": vix_change,
        "gold_change": gold_change,
        "defence_change": defence_change,
        "vix_score": vix_score,
        "gold_score": gold_score,
        "defence_score": defence_score
    }

# ---------- COMBINED PIPELINE ----------
def compute_daily_gtx():
    all_sources = []
    all_sources.extend(scrape_state_dept())
    all_sources.extend(scrape_china_mfa())
    all_sources.extend(scrape_rus_mid())
    all_sources.extend(scrape_eu_eeas())
    all_sources.extend(scrape_military_briefings())
    all_sources.extend(scrape_state_media())

    scores = []
    total_chars = 0
    sources_detail = []
    for src in all_sources:
        try:
            soup = fetch_soup(src['url'])
            text = extract_text(soup)
            if text:
                score = score_text_with_ai(text)
                if score is not None:
                    scores.append(score)
                    total_chars += len(text)
                    sources_detail.append({
                        'type': src['type'],
                        'title': src['title'],
                        'url': src['url'],
                        'chars': len(text),
                        'speaker': 'N/A'
                    })
                    logging.info(f"Scored {src['type']}: {score}")
        except Exception as e:
            logging.error(f"Error processing {src['url']}: {e}")

    if not scores:
        return None, None, []

    raw = sum(scores) / len(scores)
    if len(score_history) > 0:
        smoothed = round(sum(score_history) / len(score_history), 1)
    else:
        smoothed = round(raw, 1)
    score_history.append(raw)

    num_sources = len(sources_detail)
    if num_sources >= 4 and total_chars > 8000:
        confidence = "HIGH"
    elif num_sources >= 2 and total_chars > 3000:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return smoothed, confidence, sources_detail

# ---------- ROUTES ----------
@app.route('/health')
def health():
    return "OK"

@app.route('/ping')
def ping():
    score, confidence, sources = compute_daily_gtx()
    if score is None:
        return jsonify({"status": "error", "message": "No data"}), 500
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    return jsonify({"status": "ok", "score": score, "timestamp": ts})

@app.route('/api/gtx_latest')
def gtx_latest():
    score, confidence, sources = compute_daily_gtx()
    if score is None:
        return jsonify({"error": "No data available"}), 500
    prev = list(score_history)
    change = round(score - prev[-2], 1) if len(prev) > 1 else 0
    raw_score = prev[-1] if prev else score
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    return jsonify({
        "index": "G-Tension (GTX)",
        "score": score,
        "raw_score": raw_score,
        "change": change,
        "confidence": confidence,
        "sources": sources,
        "timestamp": ts
    })

@app.route('/api/market_gtx')
def market_gtx():
    score, label, components = compute_market_gtx()
    if score is None:
        return jsonify({"error": "Market data unavailable"}), 500
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    return jsonify({
        "index": "Market Expectation (GTX‑M)",
        "score": score,
        "label": label,
        "components": components,
        "timestamp": ts
    })

@app.route('/')
def home():
    return "G-Tension (GTX) Geopolitical Temperature Index is live. Use /api/gtx_latest"

# ---------- X AUTO‑POST ENDPOINT ----------
@app.route('/post_tweet')
def auto_post():
    try:
        score, confidence, sources = compute_daily_gtx()
        if score is None:
            return jsonify({"status": "No score available"})
        prev = list(score_history)
        change = round(score - prev[-2], 1) if len(prev) > 1 else 0
        if change == 0:
            arrow = "—"
        elif change > 0:
            arrow = f"▲{abs(change)}"
        else:
            arrow = f"▼{abs(change)}"
        if score <= 20:
            label = "Extremely Low Risk"
        elif score <= 40:
            label = "Low Risk"
        elif score <= 60:
            label = "Moderate Risk"
        elif score <= 80:
            label = "High Risk"
        else:
            label = "Extreme Risk"
        sources_count = len(sources) if sources else 0
        tweet_text = (
            f"🌍 GTX today: {score} {arrow} — {label}\n"
            f"Confidence: {confidence} | Sources: {sources_count}"
        )
        client = tweepy.Client(
            consumer_key=os.environ["X_CONSUMER_KEY"],
            consumer_secret=os.environ["X_CONSUMER_SECRET"],
            access_token=os.environ["X_ACCESS_TOKEN"],
            access_token_secret=os.environ["X_ACCESS_TOKEN_SECRET"]
        )
        response = client.create_tweet(text=tweet_text)
        logging.info(f"Tweet posted: {response.data['id']}")
        return jsonify({"status": "Tweet posted successfully"})
    except Exception as e:
        logging.error(f"Auto‑post failed: {e}")
        return jsonify({"status": f"Error posting tweet: {e}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
