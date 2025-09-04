# src/data_processing.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
from google.cloud import storage
import os
import time
import logging
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BUCKET_NAME = 'august-ai-assets-ts-sept2025'
PROJECT_ID = 'the-august-ai-coach'

# Premium curated URLs for August AI Coach
TARGET_URLS = [
    # Core Gut Health & Microbiome
    'https://www.healthline.com/nutrition/gut-microbiome-and-health',
    'https://my.clevelandclinic.org/health/body/25201-gut-microbiome',
    'https://health.clevelandclinic.org/gut-health',
    'https://health.clevelandclinic.org/how-to-improve-your-digestive-tract-naturally',
    'https://nutritionsource.hsph.harvard.edu/microbiome/',
    
    # IBS & Digestive Disorders
    'https://www.mayoclinic.org/diseases-conditions/irritable-bowel-syndrome/symptoms-causes/syc-20360016',
    'https://www.niddk.nih.gov/health-information/digestive-diseases/irritable-bowel-syndrome',
    'https://www.hopkinsmedicine.org/health/wellness-and-prevention/gut-health',
    
    # Bloating & Gas
    'https://www.niddk.nih.gov/health-information/digestive-diseases/gas-digestive-tract/symptoms-causes',
    'https://www.niddk.nih.gov/health-information/digestive-diseases/gas-digestive-tract/eating-diet-nutrition',
    'https://www.webmd.com/digestive-disorders/remedies-for-bloating',
    'https://www.hopkinsmedicine.org/health/conditions-and-diseases/gas-in-the-digestive-tract',
    
    # Harvard Medical School Sources
    'https://www.health.harvard.edu/staying-healthy/5-simple-ways-to-improve-gut-health',
    'https://www.health.harvard.edu/diseases-and-conditions/the-gut-brain-connection',
    'https://www.health.harvard.edu/nutrition/prebiotics-understanding-their-role-in-gut-health',
    
    # Gut-Brain Connection
    'https://my.clevelandclinic.org/health/body/the-gut-brain-connection',
    
    # Digestive Health Tips
    'https://www.webmd.com/digestive-disorders/digestive-health-tips',
    'https://www.webmd.com/digestive-disorders/features/bloated-bloating',
    
    # Professional Guidelines
    'https://patient.gastro.org/digestive-health-topics-a-z/',
    'https://gi.org/patients/gi-health-and-disease/',
    
    # Probiotics & Prebiotics
    'https://www.mayoclinic.org/healthy-lifestyle/consumer-health/in-depth/probiotics/art-20043198',
    'https://www.healthline.com/health/digestive-health/how-to-improve-gut-health',
    'https://www.healthline.com/nutrition/19-best-prebiotic-foods',
    
    # Comprehensive Guides
    'https://www.healthline.com/health/gut-health',
    'https://www.healthline.com/nutrition/improve-gut-bacteria'
]

class WebContentProcessor:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def scrape_content(self, url: str) -> Dict[str, str]:
        """Enhanced content scraping with better error handling and content extraction."""
        try:
            logger.info(f"Scraping: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
                element.decompose()
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Extract main content using multiple strategies
            content = ""
            
            # Strategy 1: Look for article content
            main_content = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile(r'content|article|post'))
            
            if main_content:
                paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
            else:
                # Strategy 2: Fallback to all paragraphs
                paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
            
            # Extract and clean text
            content_parts = []
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 30:  # Only meaningful content
                    content_parts.append(text)
            
            content = "\n".join(content_parts)
            
            # Clean up content
            content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive newlines
            content = re.sub(r' {2,}', ' ', content)      # Remove excessive spaces
            
            logger.info(f"Successfully scraped {len(content)} characters from {url}")
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'word_count': len(content.split()),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'word_count': 0,
                'status': f'error: {str(e)}'
            }
    
    def chunk_content(self, content: str, title: str = "", chunk_size: int = 300) -> List[Dict[str, str]]:
        """Intelligent content chunking with context preservation."""
        if not content.strip():
            return []
        
        # Split into paragraphs first
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            words = paragraph.split()
            para_length = len(words)
            
            # If adding this paragraph would exceed chunk size
            if current_length + para_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) > 50:  # Only meaningful chunks
                    chunks.append({
                        'text': chunk_text,
                        'title_context': title,
                        'word_count': current_length
                    })
                
                # Start new chunk
                current_chunk = [paragraph]
                current_length = para_length
            else:
                current_chunk.append(paragraph)
                current_length += para_length
        
        # Add remaining content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) > 50:
                chunks.append({
                    'text': chunk_text,
                    'title_context': title,
                    'word_count': current_length
                })
        
        return chunks
    
    def upload_to_gcs(self, local_path: str, gcs_path: str) -> bool:
        """Upload file to Google Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Successfully uploaded {local_path} to gs://{self.bucket_name}/{gcs_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {str(e)}")
            return False
    
    def process_all_urls(self) -> None:
        """Main processing function."""
        all_chunks = []
        processing_log = []
        
        logger.info(f"Starting to process {len(TARGET_URLS)} URLs...")
        
        for i, url in enumerate(TARGET_URLS, 1):
            logger.info(f"Processing {i}/{len(TARGET_URLS)}: {url}")
            
            # Scrape content
            result = self.scrape_content(url)
            processing_log.append(result)
            
            if result['status'] == 'success' and result['content']:
                # Chunk the content
                chunks = self.chunk_content(result['content'], result['title'])
                
                # Add metadata to chunks
                for chunk in chunks:
                    chunk.update({
                        'source_url': url,
                        'source_title': result['title'],
                        'processing_date': pd.Timestamp.now().isoformat(),
                        'domain': urlparse(url).netloc
                    })
                
                all_chunks.extend(chunks)
                logger.info(f"Generated {len(chunks)} chunks from {url}")
            
            # Be respectful to servers
            time.sleep(2)
        
        # Create DataFrame and save
        if all_chunks:
            df = pd.DataFrame(all_chunks)
            
            # Save locally first
            os.makedirs('data', exist_ok=True)
            kb_path = 'data/knowledge_base.csv'
            df.to_csv(kb_path, index=False)
            
            # Save processing log
            log_path = 'data/processing_log.json'
            with open(log_path, 'w') as f:
                json.dump(processing_log, f, indent=2)
            
            logger.info(f"Knowledge base created with {len(df)} chunks from {len([r for r in processing_log if r['status'] == 'success'])} successful sources")
            
            # Upload to GCS
            self.upload_to_gcs(kb_path, 'data/knowledge_base.csv')
            self.upload_to_gcs(log_path, 'data/processing_log.json')
            
            # Print summary
            self.print_processing_summary(processing_log, len(all_chunks))
        else:
            logger.error("No content was successfully processed!")
    
    def print_processing_summary(self, processing_log: List[Dict], total_chunks: int) -> None:
        """Print a comprehensive processing summary."""
        successful = [r for r in processing_log if r['status'] == 'success']
        failed = [r for r in processing_log if r['status'] != 'success']
        
        print("\n" + "="*80)
        print("📊 KNOWLEDGE BASE PROCESSING SUMMARY")
        print("="*80)
        print(f"✅ Successfully processed: {len(successful)}/{len(processing_log)} URLs")
        print(f"📄 Total content chunks generated: {total_chunks}")
        print(f"📝 Total words processed: {sum(r['word_count'] for r in successful):,}")
        
        if successful:
            print(f"\n🏆 TOP SOURCES BY CONTENT:")
            sorted_sources = sorted(successful, key=lambda x: x['word_count'], reverse=True)[:5]
            for i, source in enumerate(sorted_sources, 1):
                print(f"  {i}. {source['url']} ({source['word_count']:,} words)")
        
        if failed:
            print(f"\n❌ FAILED URLs ({len(failed)}):")
            for failure in failed[:5]:  # Show first 5 failures
                print(f"  • {failure['url']}: {failure['status']}")
        
        print(f"\n💾 Files saved:")
        print(f"  • data/knowledge_base.csv")
        print(f"  • data/processing_log.json")
        print(f"  • Uploaded to gs://{self.bucket_name}/data/")
        print("="*80)

if __name__ == "__main__":
    # Initialize processor
    processor = WebContentProcessor(BUCKET_NAME)
    
    # Process all URLs
    processor.process_all_urls()
    
    print("\n🎉 Phase 1 - Knowledge Base Creation Complete!")
    print("Next: Create your tone_dataset.json for personality fine-tuning")
