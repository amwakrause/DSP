import cv2
import numpy as np

def helper_check_path(p):
    return p.lower().endswith('.pdf')

# TODO: add pypdfium2 support later when we actually need pdfs
def load_input_as_image(file_path):
    if helper_check_path(file_path):
        raise NotImplementedError("pdf stuff not done yet")
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def parse_spatial_rules(symbols):
    if not symbols:
        return ""
    
    tokens = []
    med_h = np.median([s['h'] for s in symbols])
    med_y = np.median([s['y'] for s in symbols])

    i = 0
    while i < len(symbols):
        curr = symbols[i]
        
        # fraction bar check
        if curr['w'] > med_h * 2.5 and curr['h'] < med_h * 0.4:
            tokens.append(r"\frac{numerator}{denominator}")
            i += 1
            continue

        # exponent?
        if i + 1 < len(symbols):
            nxt = symbols[i + 1]
            if nxt['y'] < curr['y'] and nxt['h'] < med_h * 0.8:
                tokens.append(f"sym_{i}^{{sym_{i+1}}}")
                i += 2
                continue

        tokens.append(f"sym_{i}")
        i += 1

    return " ".join(tokens)

def preprocess_and_segment(file_path):
    img = load_input_as_image(file_path)
    if img is None:
        raise FileNotFoundError(f"bad path: {file_path}")

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbols = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 4 and h > 4:
            symbols.append({'x': x, 'y': y, 'w': w, 'h': h})

    symbols = sorted(symbols, key=lambda s: s['x'])
    return symbols

if __name__ == "__main__":
    input_file = "handwrittenequations.png" 
    
    try:
        detected_symbols = preprocess_and_segment(input_file)
        print(f"found {len(detected_symbols)} symbols")
        
        rough_latex = parse_spatial_rules(detected_symbols)
        
        words = rough_latex.split()
        print("Draft LaTeX Structure (Formatted):")
        for i in range(0, len(words), 15):
            print(" ".join(words[i:i+15]))
        
    except Exception as e:
        print(f"Error: {e}")


