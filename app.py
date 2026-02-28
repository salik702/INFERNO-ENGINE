import os
import warnings

# --- Suppress Terminal Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
warnings.filterwarnings('ignore')         

import streamlit as st
import pickle
import numpy as np
import time

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR') 
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- PAGE SETTINGS ---
st.set_page_config(
    page_title="Inferno Engine",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- HYPER-MODERN CSS STYLING ---
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stHeader"] { display: none !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem !important; 
        padding-bottom: 2rem !important;
        max-width: 1000px !important;
    }
    
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500;600;800&family=Space+Grotesk:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp {
        background-color: #050202; /* Pitch black with red undertone */
        background-image: 
            radial-gradient(ellipse at 50% 0%, rgba(220, 38, 38, 0.15) 0%, transparent 60%),
            linear-gradient(0deg, rgba(20, 5, 5, 0.8) 0%, rgba(5, 2, 2, 1) 100%),
            radial-gradient(circle at 85% 80%, rgba(249, 115, 22, 0.08) 0%, transparent 40%);
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }

    /* HUD Metrics overlay styles */
    .hud-box {
        background: rgba(20, 5, 5, 0.4);
        border: 1px solid rgba(220, 38, 38, 0.2);
        border-left: 3px solid #ef4444;
        padding: 15px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        color: #fca5a5;
        text-align: left;
        box-shadow: inset 0 0 20px rgba(220, 38, 38, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .hud-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #7f1d1d;
        margin-bottom: 5px;
    }
    
    .hud-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ef4444;
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
    }

    /* Hero Section */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    /* Glitch Animation for Title */
    @keyframes pulse-glow {
        0%, 100% { text-shadow: 0 0 20px rgba(239, 68, 68, 0.5), 0 0 40px rgba(239, 68, 68, 0.3); }
        50% { text-shadow: 0 0 30px rgba(239, 68, 68, 0.8), 0 0 60px rgba(239, 68, 68, 0.5), 0 0 10px rgba(255, 255, 255, 0.5); }
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #ffffff 0%, #fca5a5 30%, #dc2626 70%, #7f1d1d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        line-height: 1.2;
        animation: pulse-glow 3s infinite alternate;
    }
    
    /* Control Panel Elements */
    .control-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #fca5a5;
        letter-spacing: 1px;
        margin-bottom: 5px;
        text-transform: uppercase;
    }

    /* Input Field */
    .stTextInput {
        position: relative;
        width: 100% !important;
    }
    
    .stTextInput > div > div > input {
        background: rgba(15, 5, 5, 0.8) !important;
        border: 1px solid rgba(239, 68, 68, 0.2) !important;
        border-bottom: 2px solid #ef4444 !important;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 20px 20px !important;
        box-sizing: border-box !important;
        width: 100% !important;
        font-size: 1.15rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 400;
        letter-spacing: 0.5px;
        backdrop-filter: blur(15px);
        transition: all 0.3s ease;
        text-align: left; 
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .stTextInput > div > div > input:focus {
        border: 1px solid rgba(239, 68, 68, 0.8) !important;
        border-bottom: 2px solid #f97316 !important;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.3), inset 0 0 15px rgba(239, 68, 68, 0.1) !important;
        transform: translateY(-2px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #7f1d1d !important;
        text-align: left;
    }

    /* Sliders */
    .stSlider > div > div > div > div[role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid #ef4444 !important;
        box-shadow: 0 0 10px #ef4444 !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #991b1b 0%, #ef4444 100%) !important;
    }

    /* BUTTONS - GUARANTEED GREEN & PERFECT ALIGNMENT */
    div.stButton > button {
        background: linear-gradient(90deg, #166534 0%, #22c55e 50%, #10b981 100%) !important;
        color: white !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.15rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        border-radius: 8px !important;
        border: 1px solid #4ade80 !important;
        padding: 0 !important;
        min-height: 55px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        box-shadow: 0 8px 15px -5px rgba(34, 197, 94, 0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important;
        margin: 0 !important;
    }
    
    /* Rid any injected p tag margins */
    div.stButton > button p {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 1.15rem !important;
        font-weight: 800 !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 30px -5px rgba(34, 197, 94, 0.6), inset 0 1px 0 rgba(255,255,255,0.3) !important;
        background: linear-gradient(90deg, #15803d 0%, #4ade80 50%, #34d399 100%) !important;
        color: white !important;
        border-color: #22c55e !important;
    }
    
    div.stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* OUTPUT CONSOLE */
    .console-header {
        background: #110404;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-bottom: none;
        border-radius: 12px 12px 0 0;
        padding: 10px 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #ef4444;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .console-dots span {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }

    .ai-output-glass {
        background: rgba(10, 2, 2, 0.85);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-top: none;
        border-radius: 0 0 12px 12px;
        padding: 2.5rem;
        box-shadow: 0 25px 50px rgba(0,0,0,0.8), inset 0 0 30px rgba(220, 38, 38, 0.05);
        color: #f8fafc;
        font-size: 1.25rem;
        line-height: 1.8;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 400;
        min-height: 250px;
        position: relative;
        text-align: left;
    }
    
    /* Typewriter Cursor */
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
    .type-cursor {
        display: inline-block;
        width: 10px;
        height: 1.25rem;
        background: #ef4444;
        margin-left: 5px;
        vertical-align: middle;
        animation: blink 1s step-end infinite;
        box-shadow: 0 0 10px #ef4444;
    }

    .empty-state {
        color: #7f1d1d;
        font-weight: 400;
        opacity: 0.6;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Advanced Pulse Orb */
    .glow-orb {
        position: absolute;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, #dc2626 0%, transparent 70%);
        filter: blur(60px);
        opacity: 0.15;
        z-index: -1;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        animation: pulse-glow 4s infinite alternate;
    }

</style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource(show_spinner="Booting core systems...")
def load_resources():
    if not TF_AVAILABLE:
        return None, None, "TensorFlow not found. Please install tensorflow."
        
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        
        tokenizer_path = os.path.join(base_dir, 'tokenizer.pkl')
        model_path = os.path.join(base_dir, 'lstm_model.h5')
        if not os.path.exists(model_path):
             model_path = os.path.join(base_dir, 'model.h5')
        max_len_path = os.path.join(base_dir, 'max_len.pkl')

        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
            
        with open(max_len_path, 'rb') as f:
            max_len = pickle.load(f)
            
        model = load_model(model_path, compile=False)
        
        return tokenizer, model, max_len
    except Exception as e:
        return None, None, str(e)

# --- ADVANCED INFERENCE ENGINE WITH TEMPERATURE SAMPLING ---
def sample_with_temperature(preds, temperature=1.0):
    """
    Advanced text generation logic: introduces 'Temperature' (Creativity).
    Lower temperature = more rigid/predictable.
    Higher temperature = more creative/random.
    """
    preds = np.asarray(preds).astype('float64')
    
    if temperature <= 0.01:
        # Basically greedy argmax
        return np.argmax(preds)
        
    preds = np.log(preds + 1e-7) / temperature # add epsilon to avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed_text, next_words, temperature, tokenizer, model, max_len):
    result = seed_text
    
    try:
        pad_length = model.input_shape[1]
    except Exception:
        pad_length = max_len - 1 if isinstance(max_len, int) else 20
        
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=pad_length, padding='pre')
        
        predicted_probs = model.predict(token_list, verbose=0)[0] # get 1D array
        
        # Apply intelligent temperature sampling
        predicted = sample_with_temperature(predicted_probs, temperature)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
                
        if output_word == "":
            break
            
        seed_text += " " + output_word
        result += " " + output_word
        
    return result

# --- FLUID TERMINAL ANIMATION UTILS ---
def render_typing_effect(text, container):
    displayed_text = ""
    words = text.split(" ")
    
    for i, word in enumerate(words):
        displayed_text += word + " "
        cursor_html = "<span class='type-cursor'></span>" if i < len(words) - 1 else ""
        
        html_payload = f"""
        <div style="position:relative;">
            <div class="glow-orb"></div>
            <div class="console-header">
                <div>SYS.OUTPUT.STREAM</div>
                <div class="console-dots">
                    <span style="background:#ef4444;"></span>
                    <span style="background:#f97316;"></span>
                    <span style="background:#22c55e;"></span>
                </div>
            </div>
            <div class="ai-output-glass">
                <div>> {displayed_text}{cursor_html}</div>
            </div>
        </div>
        """
        container.markdown(html_payload, unsafe_allow_html=True)
        time.sleep(0.05) # Slower typing for terminal feel

# --- MAIN CONTROLLER ---
def main():
    if 'generated_text' not in st.session_state:
        st.session_state.generated_text = ""
        
    tokenizer, model, max_len = load_resources()

    # --- HERO SECTION ---
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">INFERNO ENGINE</h1>
        <p style="color: #fca5a5; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; margin-top: -10px; letter-spacing: 2px;">NEURAL SEQUENCE PREDICTION MATRIX</p>
    </div>
    """, unsafe_allow_html=True)

    if not tokenizer or not model:
        st.error("⚠️ SYSTEM FAILURE: Core weights unlinked. Missing `tokenizer.pkl`, `lstm_model.h5`, or `max_len.pkl`.", icon="🚨")
        if isinstance(max_len, str):
            st.error(f"Detailed Diagnostics: {max_len}")
        return

    # --- TOP HUD ---
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f"""
        <div class="hud-box">
            <div class="hud-label">VOCABULARY DATABASE</div>
            <div class="hud-value">{len(tokenizer.word_index):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div class="hud-box">
            <div class="hud-label">MAX TENSOR SPAN</div>
            <div class="hud-value">{model.input_shape[1] if hasattr(model, 'input_shape') else max_len}</div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown(f"""
        <div class="hud-box">
            <div class="hud-label">ENGINE STATUS</div>
            <div class="hud-value" style="color: #22c55e; text-shadow: 0 0 10px rgba(34,197,94,0.4);">ONLINE</div>
        </div>
        """, unsafe_allow_html=True)

    # --- INPUT/CONTROL LAYER ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Massive Central Prompt
    st.markdown("<div class='control-label'>[01] INJECT SEED SEQUENCE</div>", unsafe_allow_html=True)
    seed_text = st.text_input("Seed Context", placeholder="> initialize context format...", label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Advanced parameters in 2 columns
    col_params1, col_params2 = st.columns(2, gap="large")
    
    with col_params1:
        st.markdown("<div class='control-label'>[02] GENERATION HORIZON (TOKENS)</div>", unsafe_allow_html=True)
        next_words = st.slider("Length", min_value=1, max_value=200, value=50, step=1, label_visibility="collapsed")
        
    with col_params2:
        st.markdown("""
        <div class='control-label'>[03] TEMPERATURE (CREATIVITY/CHAOS)</div>
        """, unsafe_allow_html=True)
        temperature = st.slider(
            "Temperature", 
            min_value=0.1, 
            max_value=2.0, 
            value=1.0, 
            step=0.1, 
            label_visibility="collapsed",
            help="Low = Predictable. High = Chaotic/Random."
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Ignition Sequence
    btn_col1, btn_col2 = st.columns(2, gap="large")
    with btn_col1:
        generate_btn = st.button("EXECUTE IGNITION SEQUENCE ⚡")
    with btn_col2:
        if st.button("PURGE"):
            st.session_state.generated_text = ""
            st.rerun()

    # --- OUTPUT CONSOLE LAYER ---
    st.markdown("<br>", unsafe_allow_html=True)
    output_container = st.empty()
    
    if generate_btn:
        if not seed_text.strip():
            output_container.warning("⚠️ Error: Submitting empty vector to model pipeline.")
        else:
            with st.spinner(" "): 
                generated = generate_text(seed_text, next_words, temperature, tokenizer, model, max_len)
                st.session_state.generated_text = generated
            
            render_typing_effect(st.session_state.generated_text, output_container)
    else:
        if st.session_state.generated_text:
            output_container.markdown(f"""
            <div style="position:relative;">
                <div class="glow-orb"></div>
                <div class="console-header">
                    <div>SYS.OUTPUT.STREAM</div>
                    <div class="console-dots">
                        <span style="background:#ef4444;"></span>
                        <span style="background:#f97316;"></span>
                        <span style="background:#22c55e;"></span>
                    </div>
                </div>
                <div class="ai-output-glass">
                    <div>> {st.session_state.generated_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            output_container.markdown("""
            <div style="position:relative;">
                <div class="glow-orb" style="opacity: 0.05;"></div>
                <div class="console-header" style="opacity: 0.5;">
                    <div>SYS.OUTPUT.STREAM - STANDBY</div>
                    <div class="console-dots">
                        <span style="background:#ef4444; opacity:0.3;"></span>
                        <span style="background:#f97316; opacity:0.3;"></span>
                        <span style="background:#22c55e; opacity:0.3;"></span>
                    </div>
                </div>
                <div class="ai-output-glass empty-state">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">⚠️</div>
                    <div>AWAITING SEQUENCE INJECTION...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
