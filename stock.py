"""
Main entry point for the StockLyzer application
Author: Nyein Chan Ko Ko
"""

from stock_analyzer.app import StockLyzerApp

def main():
    """Main entry point for the application"""
    app = StockLyzerApp()
    app.run()

if __name__ == '__main__':
    main()
