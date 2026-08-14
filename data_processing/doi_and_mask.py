# -*- coding: utf-8 -*-
"""
Simplified PDF Processing Pipeline - DOI Extraction and Mask Creation
Author: Gerfried Millner
Version: 1.0.0
Date: 31.3.2026
Status: Development
"""
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, List
import logging
from dataclasses import dataclass
import cv2
import numpy as np
import PyPDF2
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExtractionStats:
    """Statistics for extraction results"""
    total_pdfs: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    dois_found: int = 0

class DOIExtractor:
    """Extract DOI from PDF files"""
    
    # Common DOI patterns
    DOI_PATTERNS = [
        r'doi\s*[:.]?\s*10\.\d{4,}/[^\s]+',
        r'DOI\s*[:.]?\s*10\.\d{4,}/[^\s]+',
        r'https?://doi\.org/10\.\d{4,}/[^\s]+',
        r'https?://dx\.doi\.org/10\.\d{4,}/[^\s]+',
        r'10\.\d{4,}/[^\s]+',
    ]
    
    @staticmethod
    def clean_doi(doi_text: str) -> str:
        """Clean and normalize DOI string"""
        # Remove common prefixes
        doi_text = re.sub(r'^.*?doi\s*[:.]?\s*', '', doi_text, flags=re.IGNORECASE)
        doi_text = re.sub(r'^.*?://[dx\.]*doi\.org/', '', doi_text)
        
        # Remove trailing punctuation and whitespace
        doi_text = re.sub(r'[.,;)\]\s]+$', '', doi_text)
        
        # Ensure it starts with 10.
        if not doi_text.startswith('10.'):
            return None
            
        return doi_text
    
    @staticmethod
    def extract_doi_from_text(text: str) -> Optional[str]:
        """Extract DOI from text using regex patterns"""
        for pattern in DOIExtractor.DOI_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                doi = DOIExtractor.clean_doi(match.group())
                if doi and len(doi) > 7:  # Valid DOIs should be longer than "10.xxxx"
                    return doi
        return None
    
    @staticmethod
    def extract_doi_from_pdf(pdf_path: Path) -> Optional[str]:
        """Extract DOI using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Check first few pages for DOI
                for i in range(min(3, len(reader.pages))):
                    try:
                        text = reader.pages[i].extract_text()
                        doi = DOIExtractor.extract_doi_from_text(text)
                        if doi:
                            return doi
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"PyPDF2 extraction failed: {e}")
        return None

class PDFProcessor:
    """PDF processing for DOI extraction and folder structure creation"""
    
    def save_doi_info(self, output_dir: Path, pdf_name: str, doi: Optional[str]):
        """Save DOI information to a text file in the output directory"""
        doi_file = output_dir / "doi_info.txt"
        
        with open(doi_file, 'w', encoding='utf-8') as f:
            f.write(f"PDF: {pdf_name}\n")
            if doi:
                f.write(f"DOI: {doi}\n")
                f.write(f"DOI URL: https://doi.org/{doi}\n")
            else:
                f.write("DOI: Not found\n")
            f.write(f"Extraction Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def process_pdf(self, pdf_path: Path, output_dir: Path, verbose: bool = True) -> Dict[str, any]:
        """
        Process a PDF: extract DOI and create folder structure
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        doi = DOIExtractor.extract_doi_from_pdf(pdf_path)
        if verbose:
            if doi:
                logger.info(f"📄 Found DOI: {doi}")
            else:
                logger.warning(f"📄 No DOI found in {pdf_path.name}")
        
        # Save DOI information
        self.save_doi_info(output_dir, pdf_path.name, doi)
        
        if verbose:
            logger.info(f"📄 DOI extraction complete for {pdf_path.name}")
        
        return {
            "success": True,
            "output_dir": output_dir,
            "doi": doi,
        }
    
    def process_folder(self, pdf_folder: Path, output_dir: Path, verbose: bool = True) -> ExtractionStats:
        """Process all PDFs in a folder"""
        pdf_files = list(pdf_folder.glob("*.pdf"))
        stats = ExtractionStats()
        stats.total_pdfs = len(pdf_files)
        
        logger.info(f"📁 Found {len(pdf_files)} PDF files to process")
        logger.info("🔍 DOI extraction mode (no figure extraction)")
        
        # Create a master DOI log file
        master_doi_log = output_dir / "master_doi_log.txt"
        
        with open(master_doi_log, 'w', encoding='utf-8') as master_log:
            master_log.write("PDF DOI Extraction Log\n")
            master_log.write("=" * 50 + "\n")
            master_log.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for pdf_file in pdf_files:
            # Use author-year format for folder names
            author_year = pdf_file.stem.split('_')[0] if '_' in pdf_file.stem else pdf_file.stem[:20]
            paper_output_dir = output_dir / author_year
            
            if verbose:
                logger.info(f"Processing {pdf_file.name} -> output to {author_year}/")
            
            result = self.process_pdf(pdf_file, paper_output_dir, verbose)
            
            # Update master DOI log
            with open(master_doi_log, 'a', encoding='utf-8') as master_log:
                master_log.write(f"PDF: {pdf_file.name}\n")
                master_log.write(f"Output Folder: {author_year}\n")
                if result.get("doi"):
                    master_log.write(f"DOI: {result['doi']}\n")
                    master_log.write(f"DOI URL: https://doi.org/{result['doi']}\n")
                    stats.dois_found += 1
                else:
                    master_log.write("DOI: Not found\n")
                master_log.write("-" * 30 + "\n\n")
            
            if result["success"]:
                stats.successful_extractions += 1
            else:
                stats.failed_extractions += 1
                self._log_failure(output_dir, pdf_file.name, result.get("error", "Unknown error"))
        
        self._print_extraction_summary(stats)
        return stats
    
    def _log_failure(self, output_dir: Path, pdf_name: str, error: str):
        """Log extraction failures"""
        log_file = output_dir / "extraction_failures.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{pdf_name}: {error}\n")
    
    def _print_extraction_summary(self, stats: ExtractionStats):
        """Print extraction statistics"""
        logger.info("=" * 60)
        logger.info("DOI EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total PDFs processed: {stats.total_pdfs}")
        logger.info(f"Successful processing (no runtime error): {stats.successful_extractions}")
        logger.info(f"Processing failures (errors): {stats.failed_extractions}")
        logger.info(f"DOIs extracted: {stats.dois_found}")
        
        if stats.total_pdfs > 0:
            success_rate = (stats.successful_extractions / stats.total_pdfs) * 100
            doi_rate = (stats.dois_found / stats.total_pdfs) * 100
            logger.info(f"Processing success rate: {success_rate:.1f}%")
            logger.info(f"DOI extraction rate: {doi_rate:.1f}%")
        logger.info("=" * 60)

def extract_dois_from_pdfs(pdf_folder: str, output_dir: str):
    """
    Extract DOIs from PDFs in a folder and create organized folder structure
    
    Args:
        pdf_folder: Folder containing PDF files
        output_dir: Output directory for organized folders and DOI info
    """
    pdf_folder = Path(pdf_folder)
    output_dir = Path(output_dir)
    
    logger.info("=" * 60)
    logger.info("EXTRACTING DOI INFORMATION FROM PDFs")
    logger.info("=" * 60)
    
    processor = PDFProcessor()
    stats = processor.process_folder(pdf_folder, output_dir)
    
    logger.info(f"\n📁 Results saved to: {output_dir}")
    logger.info(f"📄 Master DOI log: {output_dir / 'master_doi_log.txt'}")
    
    return stats

def build_mask(
    image: np.ndarray,
    dilate_px: int = 4,
    save_path: str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple function to mask bright colored regions that were manually marked in GIMP.
    
    Args:
        image: Input image with bright colored regions marking areas to ignore
        dilate_px: Dilation size for expanding masked regions
        save_path: path to location where mask is stored as numpy array, default is None
    Returns:
        preview: Image with masked regions set to white
        mask: Binary mask (255=ignore, 0=keep)
    """
    # Ensure BGR
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        was_gray = True
    else:
        bgr = image.copy()
        was_gray = False

    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Detect bright colored regions (what you marked in GIMP)
    color_ranges = [
        ((0,   80,  80), (15, 255, 255)),   # red
        ((165, 80,  80), (180,255, 255)),   # red wrap
        ((85,  80,  80), (130,255, 255)),   # blue  
        ((35,  80,  80), (85, 255, 255)),   # green
        ((15,  80,  80), (35, 255, 255)),   # yellow/orange
        ((130, 80,  80), (165,255, 255)),   # magenta/purple
        ((85,  80,  80), (100,255, 255)),   # cyan
    ]
    
    # Combine all colored regions
    for lo, hi in color_ranges:
        color_mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.bitwise_or(mask, color_mask)
    
    # Optional: dilate to ensure complete coverage
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Clean up small noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)
    
    # Create preview
    preview = bgr.copy()
    preview[mask > 0] = (255, 255, 255)
    
    if was_gray:
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        
    if save_path is not None:
        np.save(save_path, mask)
    
    return preview, mask

def build_mask_from_folder(images_folder: str, dilate_px: int = 4, save_preview: bool = False) -> List[np.ndarray]:
    """
    Extract masks from images in a folder and save them in a subfolder.
    
    Args:
        images_folder: Folder containing image files to extract masks from
        dilate_px: Dilation size for expanding masked regions, default is 4 pixels
        save_preview: If True, saves a preview image with masked regions in white
    Returns:
        list_masks: List of masks for each processed image
    """
    logger.info("=" * 60)
    logger.info("EXTRACTING MASKS FROM IMAGES")
    logger.info("=" * 60)
    
    saved_path_folder = os.path.join(images_folder, "masks")
    if not os.path.exists(saved_path_folder):
        os.makedirs(saved_path_folder)

    list_masks = []
    for image_file in os.listdir(images_folder):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue
        
        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(saved_path_folder, f"{os.path.splitext(image_file)[0]}_mask.npy")
        
        # Build mask
        preview, mask = build_mask(cv2.imread(image_path), dilate_px=dilate_px, save_path=mask_path)
        
        if save_preview:
            preview_path = os.path.join(saved_path_folder, f"{os.path.splitext(image_file)[0]}_preview.png")
            cv2.imwrite(preview_path, preview)
        
        list_masks.append(mask)

    logger.info(f"\n📁 Extracted masks and saved to: {saved_path_folder}")
    
    return list_masks

def build_masks_from_subfolders(root_folder: str, dilate_px: int = 4, save_preview: bool = False):
    """
    Build masks for images in all subfolders of a root folder.
    
    Args:
        root_folder: Root folder containing subfolders with images
        dilate_px: Dilation size for expanding masked regions, default is 4 pixels
        save_preview: If True, saves preview images with masked regions in white
    """
    root_path = Path(root_folder)
    subfolders = [f for f in root_path.iterdir() if f.is_dir()]
    
    logger.info(f"Found {len(subfolders)} subfolders to process")
    
    for subfolder in tqdm(subfolders, desc="Building masks for subfolders"):
        try:
            logger.info(f"Processing: {subfolder.name}")
            build_mask_from_folder(
                images_folder=str(subfolder),
                dilate_px=dilate_px,
                save_preview=save_preview
            )
        except Exception as e:
            logger.error(f"Failed to process {subfolder.name}: {e}")
    
    logger.info("Finished processing all subfolders")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "doi":
            # python script.py doi <pdf_folder> <output_folder>
            if len(sys.argv) >= 4:
                extract_dois_from_pdfs(sys.argv[2], sys.argv[3])
            else:
                print("Usage: python script.py doi <pdf_folder> <output_folder>")

        elif command in ("mask", "build-mask"):
            # python script.py mask <images_folder> [dilate_px]
            if len(sys.argv) >= 3:
                images_folder = sys.argv[2]
                dilate_px = int(sys.argv[3]) if len(sys.argv) >= 4 else 4
                masks = build_mask_from_folder(images_folder, dilate_px=dilate_px, save_preview=False)
                print(f"Processed {len(masks)} images (dilate_px={dilate_px}). "
                      f"Masks saved to '{os.path.join(images_folder, 'masks')}'.")
            else:
                print("Usage: python script.py mask <images_folder> [dilate_px]")

        elif command in ("mask-preview", "preview-mask", "mask+preview"):
            # python script.py mask-preview <images_folder> [dilate_px]
            if len(sys.argv) >= 3:
                images_folder = sys.argv[2]
                dilate_px = int(sys.argv[3]) if len(sys.argv) >= 4 else 4
                masks = build_mask_from_folder(images_folder, dilate_px=dilate_px, save_preview=True)
                print(f"Processed {len(masks)} images (dilate_px={dilate_px}, preview=on). "
                      f"Masks & previews saved to '{os.path.join(images_folder, 'masks')}'.")
            else:
                print("Usage: python script.py mask-preview <images_folder> [dilate_px]")

        elif command == "mask-all":
            # python script.py mask-all <root_folder> [dilate_px] [save_preview]
            if len(sys.argv) >= 3:
                root_folder = sys.argv[2]
                dilate_px = int(sys.argv[3]) if len(sys.argv) >= 4 else 4
                save_preview = sys.argv[4].lower() == 'true' if len(sys.argv) >= 5 else False
                build_masks_from_subfolders(root_folder, dilate_px=dilate_px, save_preview=save_preview)
            else:
                print("Usage: python script.py mask-all <root_folder> [dilate_px] [save_preview]")

        else:
            print("Available commands:")
            print("  doi <pdf_folder> <output_folder>                    - Extract DOIs from PDFs")
            print("  mask <images_folder> [dilate_px]                    - Build masks for images in folder")
            print("  mask-preview <images_folder> [dilate_px]            - Build masks and save preview PNGs")
            print("  mask-all <root_folder> [dilate_px] [save_preview]   - Build masks for all subfolders")
    else:
        print("Usage:")
        print("  python script.py doi <pdf_folder> <output_folder>")
        print("  python script.py mask <images_folder> [dilate_px]")
        print("  python script.py mask-preview <images_folder> [dilate_px]")
        print("  python script.py mask-all <root_folder> [dilate_px] [save_preview]")
        
        module_path = str.rsplit(os.path.abspath(''), os.path.sep, 1)[0]
        sys.path.append(module_path)
        
        #######################################################################
        #######################################################################
        #######################################################################
        
        # path_images_folder = os.path.join(module_path, "data_extraction", "extracted_figures")
        # path_images_folder = os.path.join(module_path, "data_extraction", "processed_images")
        # path_pdfs = os.path.join(module_path, "data_extraction", "pdfs")
        
        path_images_folder = module_path + r'\data\processed_images'
        path_pdfs = module_path + r'\data\pdf'
        
        # # # Extract DOIs from PDFs
        # extract_dois_from_pdfs(
        #     pdf_folder=path_pdfs,
        #     output_dir=path_images_folder 
        # )
        
        # # Create masks for all subfolders
        # build_masks_from_subfolders(path_images_folder, dilate_px=4, save_preview=True)
