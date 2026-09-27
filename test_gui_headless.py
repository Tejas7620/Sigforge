import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from sigforge.gui.worker import AnalysisWorker

def test_worker():
    app = QApplication(sys.argv)
    
    worker = AnalysisWorker("test_qpsk.bin")
    
    def on_progress(val, msg):
        print(f"Progress: {val}% - {msg}")
        
    def on_stage(stage, status):
        print(f"Stage: {stage} -> {status}")
        
    def on_finished(res):
        print("Finished successfully")
        app.quit()
        
    def on_error(msg):
        print(f"Error: {msg}")
        app.quit()
        
    worker.progress.connect(on_progress)
    worker.stage_update.connect(on_stage)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    
    worker.start()
    
    QTimer.singleShot(10000, app.quit) # Timeout after 10s
    app.exec()
    
if __name__ == "__main__":
    test_worker()
