import json
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import pandas as pd
import os

class HTMLCleaner:
    def __init__(self):
        self.unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            'advertisement', 'ads', 'social', 'comment', 'popup',
            'modal', 'cookie', 'newsletter', 'subscription', 'noscript',
            'iframe', 'embed', 'object'
        ]
        
        self.unwanted_patterns = [
            'nav', 'menu', 'sidebar', 'footer', 'header', 'ad',
            'advertisement', 'social', 'share', 'comment', 'related',
            'popup', 'modal', 'cookie', 'newsletter', 'subscription',
            'breadcrumb', 'pagination', 'widget', 'banner'
        ]
    
    def clean_html_content(self, html_content, url=None):
        """Clean HTML content and extract main text"""
        if not html_content:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted tags
            for tag in self.unwanted_tags:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Remove elements by class/id patterns
            for pattern in self.unwanted_patterns:
                # Remove by class
                for element in soup.find_all(class_=re.compile(pattern, re.I)):
                    element.decompose()
                # Remove by id
                for element in soup.find_all(id=re.compile(pattern, re.I)):
                    element.decompose()
            
            # Focus on main content areas
            main_content = self.find_main_content(soup)
            
            if main_content:
                soup = main_content
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up text
            text = self.clean_text(text)
            
            return text
            
        except Exception as e:
            print(f"Error cleaning HTML: {e}")
            return ""
    
    def find_main_content(self, soup):
        """Find the main content area"""
        content_selectors = [
            'main', 'article', '[role="main"]', '.main-content',
            '.content', '.post-content', '.article-content', 
            '.entry-content', '.page-content', '.text-content'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                return main_content
        
        # If no main content found, try to find largest text block
        all_divs = soup.find_all('div')
        if all_divs:
            largest_div = max(all_divs, key=lambda div: len(div.get_text()))
            if len(largest_div.get_text()) > 200:
                return largest_div
        
        return soup
    
    def clean_text(self, text):
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove common boilerplate phrases
        boilerplate_patterns = [
            r'cookie policy.*?accept',
            r'privacy policy.*?continue',
            r'subscribe.*?newsletter',
            r'follow us on.*?social',
            r'share this.*?article',
            r'click here.*?more'
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def is_quality_content(self, text, min_length=100, max_length=50000):
        """Check if content meets quality criteria"""
        if not text or len(text) < min_length or len(text) > max_length:
            return False
        
        words = text.split()
        if len(words) < 20:
            return False
        
        # Check for spam (too many repeated words)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        max_word_freq = max(word_freq.values()) if word_freq else 0
        if max_word_freq > len(words) * 0.1:
            return False
        
        # Check for navigation-heavy content
        nav_indicators = ['home', 'about', 'contact', 'privacy', 'terms', 'login', 'register']
        nav_count = sum(1 for word in words[:50] if word.lower() in nav_indicators)
        if nav_count > 5:
            return False
        
        return True
    
    def process_dataset(self, input_file, output_file):
        """Process the entire JSONL dataset"""
        print(f"Starting HTML cleaning for {input_file}...")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"❌ Error: Input file '{input_file}' not found!")
            return []
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Created output directory: {output_dir}")
        
        cleaned_data = []
        total_processed = 0
        quality_filtered = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    total_processed += 1
                    
                    # Get the text content (adjust key names based on your data)
                    raw_text = data.get('text', '') or data.get('content', '')
                    url = data.get('url', '')
                    
                    # Clean the content
                    cleaned_content = self.clean_html_content(raw_text, url)
                    
                    # Check quality
                    if self.is_quality_content(cleaned_content):
                        quality_filtered += 1
                        
                        # Add cleaned content and metadata
                        data['cleaned_text'] = cleaned_content
                        data['content_length'] = len(cleaned_content)
                        data['word_count'] = len(cleaned_content.split())
                        data['paragraph_count'] = len([p for p in cleaned_content.split('\n\n') if p.strip()])
                        
                        # Extract domain
                        if url:
                            domain = urlparse(url).netloc
                            data['domain'] = domain
                        
                        cleaned_data.append(data)
                    
                    if line_num % 100 == 0:
                        print(f"Processed {line_num} items... ({quality_filtered} quality items)")
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        # Save cleaned data
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n✅ Cleaning completed!")
        print(f"📊 Stats:")
        print(f"  - Total processed: {total_processed}")
        print(f"  - Quality filtered: {quality_filtered}")
        print(f"  - Success rate: {quality_filtered/total_processed*100:.1f}%")
        print(f"  - Output saved to: {output_file}")
        
        return cleaned_data
    
    def analyze_results(self, data):
        """Analyze the cleaned dataset"""
        if not data:
            print("No data to analyze")
            return
        
        df = pd.DataFrame(data)
        
        print(f"\n📈 Dataset Analysis:")
        print(f"  - Total items: {len(df)}")
        print(f"  - Avg content length: {df['content_length'].mean():.0f} chars")
        print(f"  - Avg word count: {df['word_count'].mean():.0f} words")
        
        print(f"\n📏 Content length distribution:")
        print(df['content_length'].describe())
        
        if 'domain' in df.columns:
            print(f"\n🌐 Top domains:")
            print(df['domain'].value_counts().head(10))
        
        # Show sample cleaned content
        print(f"\n📄 Sample cleaned content:")
        for i in range(min(2, len(df))):
            print(f"\n--- Sample {i+1} ---")
            print(f"URL: {df.iloc[i].get('url', 'N/A')}")
            print(f"Length: {df.iloc[i]['content_length']} chars")
            print(f"Preview: {df.iloc[i]['cleaned_text'][:300]}...")

def main():
    """Main function to run the HTML cleaning process"""
    
    # Initialize cleaner
    cleaner = HTMLCleaner()
    
    # Updated file paths for organized structure
    input_file = 'data/raw/cds.jsonl'      # Input from raw data folder
    output_file = 'data/processed/cds_cleaned.jsonl'  # Output to processed folder
    
    # Check if we have the input file
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Available files in data/raw/:")
        if os.path.exists('data/raw/'):
            for file in os.listdir('data/raw/'):
                if file.endswith('.jsonl'):
                    print(f"  - {file}")
        return
    
    print(f"🔍 Processing: {input_file}")
    print(f"💾 Output will be saved to: {output_file}")
    
    # Clean the data
    cleaned_data = cleaner.process_dataset(input_file, output_file)
    
    # Analyze results
    cleaner.analyze_results(cleaned_data)

if __name__ == "__main__":
    main()