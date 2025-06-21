import os
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
import logging
from datetime import datetime
import re

class PDFProcessor:
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Setup paths using absolute paths for reliability
        current_dir = Path(os.getcwd())
        self.base_dir = current_dir
        self.input_dir = self.base_dir / 'data' / 'raw' / 'cv' / 'test'
        self.output_dir = self.base_dir / 'data' / 'processed' / 'cv'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")

    # ... giữ nguyên các phương thức khác ...

    def save_results(self, results: list):
        """Save processing results in both JSON and CSV formats with NULL handling"""
        try:
            if not results:
                self.logger.warning("No results to save")
                return

            # Create DataFrame
            df = pd.DataFrame(results)

            # Log số lượng NULL trước khi xử lý
            self.logger.info("NULL values before cleaning:")
            self.logger.info(df[['education', 'experience', 'skills']].isnull().sum())

            # Tạo file chứa các dòng NULL trước khi xóa
            null_rows = df[df[['education', 'experience', 'skills']].isnull().any(axis=1)]
            if not null_rows.empty:
                null_path = self.output_dir / 'null_entries.json'
                null_rows.to_json(null_path, orient='records', indent=2)
                self.logger.info(f"Saved {len(null_rows)} NULL entries to {null_path}")

            # Drop các dòng có NULL trong 3 cột chính
            df_cleaned = df.dropna(subset=['education', 'experience', 'skills'])
            
            # Drop các dòng có giá trị rỗng
            df_cleaned = df_cleaned[
                (df_cleaned['education'].str.strip() != '') & 
                (df_cleaned['experience'].str.strip() != '') & 
                (df_cleaned['skills'].str.strip() != '')
            ]

            # Log số lượng dữ liệu đã xử lý
            self.logger.info(f"Original records: {len(df)}")
            self.logger.info(f"Records after cleaning: {len(df_cleaned)}")
            self.logger.info(f"Removed records: {len(df) - len(df_cleaned)}")

            # Save cleaned data as JSON
            json_path = self.output_dir / 'processed_results.json'
            df_cleaned.to_json(json_path, orient='records', indent=2)
            self.logger.info(f"Cleaned results saved to {json_path}")

            # Save cleaned data as CSV
            csv_path = self.output_dir / 'cv_sections.csv'
            df_cleaned.to_csv(csv_path, index=False)
            self.logger.info(f"Cleaned results saved to {csv_path}")

            # Save cleaning statistics
            stats = {
                'original_count': len(df),
                'cleaned_count': len(df_cleaned),
                'removed_count': len(df) - len(df_cleaned),
                'null_counts_per_column': df[['education', 'experience', 'skills']].isnull().sum().to_dict(),
                'processed_date': datetime.now().isoformat()
            }
            stats_path = self.output_dir / 'cleaning_stats.json'
            pd.DataFrame([stats]).to_json(stats_path, orient='records', indent=2)
            self.logger.info(f"Cleaning statistics saved to {stats_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

def main():
    try:
        # Initialize processor
        processor = PDFProcessor()

        # Process PDFs
        results = processor.process_all_pdfs()

        # Save results with NULL handling
        processor.save_results(results)

        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Total PDFs processed: {len(results)}")
        if results:
            print(f"Results saved to: {processor.output_dir}")
            print(f"Check 'cleaning_stats.json' for detailed cleaning information")
        else:
            print("No documents were processed successfully")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()

