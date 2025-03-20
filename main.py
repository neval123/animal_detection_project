import tkinter as tk
from tkinter import filedialog
import os
from animal_detector import AnimalDetector


class AnimalDetectorUI:
    def __init__(self, master):
        # Konfiguracja okna
        self.master = master
        self.master.title("Animal Detector")
        self.master.geometry("400x200")
        self.master.resizable(False, False)

        # Model, parametr w postaci ścieżki do wytrenowanego modelu
        self.detector = AnimalDetector('./runs/train/exp3/weights/best.pt')

        self.image_path = None

        # Przycisk do wyboru obrazu
        self.select_button = tk.Button(
            self.master,
            text="Wybierz obraz",
            command=self.select_image
        )
        self.select_button.pack(pady=10)

        # Label wypisujący ścieżkę do wybranego obrazu
        self.file_path_label = tk.Label(self.master, text="Brak wybranego pliku")
        self.file_path_label.pack(pady=5)

        # Przycisk do odpalenia detekcji
        self.detect_button = tk.Button(
            self.master,
            text="Detektuj zwierzęta",
            command=self.detect_animals
        )
        self.detect_button.pack(pady=10)

        # Label wypisujący ścieżkę do zapisanego po detekcji obrazu
        self.result_label = tk.Label(self.master, text="")
        self.result_label.pack(pady=5)


    # Metoda do wyboru obrazu z dysku
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.file_path_label.config(text=f"Wybrany plik:\n{self.image_path}")
            print(f"Wybrano plik: {self.image_path}")



    def detect_animals(self):
        if not self.image_path:
            print("Najpierw wybierz obraz!")
            return

        # Metoda detekcjii
        detections = self.detector.detect_animals(self.image_path, save_results=True)

        # Ścieżka do zapisu obrazu po detekcji
        base, ext = os.path.splitext(os.path.basename(self.image_path))
        output_path = os.path.join("detections", f"{base}_after_detection{ext}")

        # Metoda rysowania detekcji na obrazie
        self.detector.draw_detections(self.image_path, detections, output_path)

        if detections:
            for det in detections:
                print(f"- {det['class']}: {det['confidence']:.2f}")
            print(f"Zapisano obraz z detekcjami: {output_path}")
            self.result_label.config(text=f"Zapisano obraz z detekcjami w katalogu projektu:\n{output_path}")
        else:
            print("Nie wykryto żadnych zwierząt.")

# Uruchomienie aplikacji
def main():
    root = tk.Tk()
    app = AnimalDetectorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()