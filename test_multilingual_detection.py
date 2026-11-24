from core.transcriber import analyze_segment_language

samples = [
    ("Today we have a sprint planning meeting to review Q4 deliverables.", "en", 0.99),
    ("आज हामी नयाँ फिचरको बारेमा छलफल गर्दैछौं, काम समयमा सम्पन्न गर्नुपर्छ।", "ne", 0.95),
    ("आज की मीटिंग में हम नए फीचर्स पर चर्चा करेंगे और काम समय पर पूरा करना है।", "hi", 0.98),
    ("हाम्रो sprint goal धेरै important छ, please deploy today before 5pm.", "ne", 0.88),
    ("कल का deployment timeline discuss करेंगे, please update the ticket.", "hi", 0.92),
    ("हामीले meeting मा decide गर्यौँ कि backend API update गर्नुपर्छ।", "ne", 0.91),
]

print("=== Multilingual Lexical & Code-Switching Detection Test ===")
for text, whisper_code, prob in samples:
    res = analyze_segment_language(text, whisper_code, prob)
    disp = res["display_name"]
    code = res["primary_code"]
    cs = res["is_code_switched"]
    print(f"Text: {text}")
    print(f"  -> Whisper Code: {whisper_code} | Detected: {disp} (Code: {code}, Code-switched: {cs})")
