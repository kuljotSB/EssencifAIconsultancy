try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
except ImportError:
    raise ImportError("Diese Anwendung benötigt tkinter und ttkbootstrap. Installieren Sie ttkbootstrap mit 'pip install ttkbootstrap'.")

from PyPDF2 import PdfReader, PdfWriter
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import os
import json
from functools import partial

class PDFSplitterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KIEBIDZ Belegsplitter")
        
        # Grundlegende Variablen
        self.pdf_path = None
        self.split_points = set()
        self.thumbnail_scale = 0.3  # Starten mit 30% Zoom
        self.selected_index = 0
        self.saved_split_path = None
        self.doc = None
        self.thumbnails = []
        self.separator_buttons = []

        # Warte einen Moment, bis das Fenster vollständig initialisiert ist
        self.root.update_idletasks()

        # Hauptlayout erstellen
        self.create_layout()
        
        # Event-Binding
        self.root.bind("<Key>", self.handle_keypress)
        
        # UI-Elemente initialisieren
        self.update_ui_state()

    def create_layout(self):
        """Erstellt das Hauptlayout der Anwendung"""
        # Hauptcontainer: Links PDF-Vorschau, rechts Steuerelemente
        self.main_paned = ttk.PanedWindow(self.root, orient=HORIZONTAL)
        self.main_paned.pack(fill=BOTH, expand=YES)

        # Linke Seite: PDF-Vorschau
        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=32)  # Mehr Platz für PDF-Vorschau
        
        # Toolbar für Zoom-Kontrollen
        self.create_toolbar()
        
        # Scrollbarer Bereich für Thumbnails
        self.create_thumbnail_area()

        # Rechte Seite: Steuerelemente
        self.right_frame = ttk.Frame(self.main_paned, bootstyle="light")
        self.main_paned.add(self.right_frame, weight=1)  # Schmalerer rechter Bereich
        
        # Header mit Titel und Logo
        self.create_header()
        
        # Aktionsbereich
        self.create_actions_area()
        
        # Buttons unten
        self.create_bottom_buttons()
        
        # Statusleiste
        self.create_statusbar()

    def create_toolbar(self):
        """Erstellt die Toolbar mit Zoom-Kontrollen"""
        self.toolbar = ttk.Frame(self.left_frame)
        self.toolbar.pack(fill=X, pady=5, padx=10)
        
        self.zoom_out_btn = ttk.Button(
            self.toolbar, 
            text="−", 
            bootstyle="secondary-outline", 
            command=self.zoom_out
        )
        self.zoom_out_btn.pack(side=LEFT, padx=5)
        
        self.zoom_in_btn = ttk.Button(
            self.toolbar, 
            text="+", 
            bootstyle="secondary-outline", 
            command=self.zoom_in
        )
        self.zoom_in_btn.pack(side=LEFT, padx=5)
        
        self.zoom_reset_btn = ttk.Button(
            self.toolbar, 
            text="100%", 
            bootstyle="secondary-outline", 
            command=self.zoom_reset
        )
        self.zoom_reset_btn.pack(side=LEFT, padx=5)
        
        self.status_label = ttk.Label(self.toolbar, text="Keine Datei geladen")
        self.status_label.pack(side=RIGHT, padx=10)

    def create_thumbnail_area(self):
        """Erstellt den scrollbaren Bereich für Thumbnails"""
        # Container für den Canvas
        self.canvas_frame = ttk.Frame(self.left_frame)
        self.canvas_frame.pack(fill=BOTH, expand=YES, padx=10, pady=10)
        
        # Canvas zum Scrollen
        self.canvas = tk.Canvas(self.canvas_frame, bg="#f0f0f0", highlightthickness=0)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=YES)
        
        # Scrollbar hinzufügen
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient=VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Mausrad-Scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Container für Thumbnails im Canvas
        self.thumbnail_container = ttk.Frame(self.canvas)
        self.thumbnail_window = self.canvas.create_window(
            (0, 0), 
            window=self.thumbnail_container, 
            anchor="nw",
            width=self.canvas.winfo_width()
        )
        
        # Konfiguriere das Thumbnail-Container-Frame für Grid-Layout
        for i in range(4):  # Vorkonfigurieren für max. 4 Spalten
            self.thumbnail_container.columnconfigure(i, weight=1)
        
        # Wenn sich die Größe des Containers ändert, aktualisiere die Scrollregion
        self.thumbnail_container.bind("<Configure>", self._configure_thumbnail_area)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_mousewheel(self, event):
        """Scrollt mit dem Mausrad"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _configure_thumbnail_area(self, event=None):
        """Aktualisiert die Scrollregion, wenn sich die Größe des Containers ändert"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        """Aktualisiert die Breite des Thumbnail-Containers, wenn sich die Canvas-Größe ändert"""
        if event:
            # Passe die Breite des Thumbnail-Containers an die Canvas-Breite an
            self.canvas.itemconfig(self.thumbnail_window, width=event.width)

    def create_header(self):
        """Erstellt den Header-Bereich mit Titel und Logo"""
        self.header_frame = ttk.Frame(self.right_frame, bootstyle="light")
        self.header_frame.pack(pady=20, fill=X)
        
        self.title_label = ttk.Label(
            self.header_frame, 
            text="KIEBIDZ", 
            font=("Segoe UI", 24, "bold"), 
            bootstyle="primary"
        )
        self.title_label.pack()
        
        self.subtitle_label = ttk.Label(
            self.header_frame, 
            text="Belegsplitter", 
            font=("Segoe UI", 16),
            bootstyle="primary"
        )
        self.subtitle_label.pack()
        
        self.version_label = ttk.Label(
            self.header_frame, 
            text="v1.0", 
            font=("Segoe UI", 10),
            bootstyle="secondary"
        )
        self.version_label.pack(pady=(0, 10))
        
        # Trennlinie
        ttk.Separator(self.right_frame, bootstyle="primary").pack(fill=X, padx=20, pady=10)

    def create_actions_area(self):
        """Erstellt den Aktionsbereich mit Laden-Button und Hilfetext"""
        self.actions_frame = ttk.Frame(self.right_frame, bootstyle="light")
        self.actions_frame.pack(pady=10, fill=BOTH, expand=YES, padx=20)
        
        self.action_heading = ttk.Label(
            self.actions_frame, 
            text="AKTIONEN", 
            font=("Segoe UI", 12, "bold"),
            bootstyle="secondary"
        )
        self.action_heading.pack(anchor=W, pady=(0, 10))
        
        # Laden-Button
        self.load_button = ttk.Button(
            self.actions_frame, 
            text="PDF laden", 
            command=self.load_pdf, 
            bootstyle="primary", 
            width=20
        )
        self.load_button.pack(fill=X, pady=5)
        
        # Progressbar für Ladevorgang
        self.progress = ttk.Progressbar(
            self.actions_frame, 
            bootstyle="primary-striped",
            mode="determinate",
            value=0
        )
        self.progress.pack(fill=X, pady=5)
        self.progress.pack_forget()  # Ausblenden bis benötigt
        
        # Hilfetext
        self.help_text = ttk.Label(
            self.actions_frame,
            text="Trennungen mit Leertaste oder durch Klicken setzen/entfernen. Navigation mit Pfeiltasten möglich.",
            wraplength=250,
            justify=LEFT,
            bootstyle="secondary"
        )
        self.help_text.pack(fill=X, pady=10)

    def create_bottom_buttons(self):
        """Erstellt die unteren Buttons für Speichern und PDF-Erzeugung"""
        self.bottom_frame = ttk.Frame(self.right_frame, bootstyle="light")
        self.bottom_frame.pack(side=BOTTOM, fill=X, padx=20, pady=20)
        
        self.split_button = ttk.Button(
            self.bottom_frame, 
            text="Trennungen speichern", 
            command=self.save_split_points, 
            bootstyle="success-outline",
            width=20
        )
        self.split_button.pack(fill=X, pady=5)
        
        self.generate_button = ttk.Button(
            self.bottom_frame, 
            text="PDFs erzeugen", 
            command=self.generate_split_pdfs, 
            bootstyle="success",
            width=20
        )
        self.generate_button.pack(fill=X, pady=5)

    def create_statusbar(self):
        """Erstellt die Statusleiste am unteren Rand"""
        self.statusbar = ttk.Frame(self.root)
        self.statusbar.pack(side=BOTTOM, fill=X)
        
        self.status_text = ttk.Label(
            self.statusbar, 
            text="Bereit", 
            bootstyle="secondary"
        )
        self.status_text.pack(side=LEFT, padx=10, pady=2)

    def update_ui_state(self):
        """Aktualisiert den Zustand der UI-Elemente basierend auf dem aktuellen Programmzustand"""
        has_pdf = self.pdf_path is not None
        has_separators = len(self.separator_buttons) > 0
        
        # Aktiviere/Deaktiviere Buttons basierend auf Zustand
        self.split_button.configure(state=NORMAL if has_pdf else DISABLED)
        self.generate_button.configure(state=NORMAL if has_pdf else DISABLED)
        self.zoom_in_btn.configure(state=NORMAL if has_pdf else DISABLED)
        self.zoom_out_btn.configure(state=NORMAL if has_pdf else DISABLED)
        self.zoom_reset_btn.configure(state=NORMAL if has_pdf else DISABLED)
        
        # Status aktualisieren
        if has_pdf:
            filename = os.path.basename(self.pdf_path)
            self.status_label.configure(text=f"Datei: {filename}")
            
            if self.doc:
                page_count = len(self.doc)
                split_count = len(self.split_points)
                self.status_text.configure(
                    text=f"{page_count} Seiten | {split_count} Trennungen | Zoom: {int(self.thumbnail_scale * 100)}%"
                )
        else:
            self.status_label.configure(text="Keine Datei geladen")
            self.status_text.configure(text="Bereit")

    def zoom_in(self):
        """Vergrößert die Thumbnail-Ansicht"""
        self.thumbnail_scale = min(1.0, self.thumbnail_scale + 0.1)
        self.reload_thumbnails()
    
    def zoom_out(self):
        """Verkleinert die Thumbnail-Ansicht"""
        self.thumbnail_scale = max(0.1, self.thumbnail_scale - 0.1)
        self.reload_thumbnails()
    
    def zoom_reset(self):
        """Setzt den Zoom zurück auf 30%"""
        self.thumbnail_scale = 0.3
        self.reload_thumbnails()

    def handle_keypress(self, event):
        """Behandelt Tastaturereignisse"""
        if not self.separator_buttons:
            return

        max_index = len(self.separator_buttons) - 1

        if event.keysym == "Left":
            self.selected_index = max(0, self.selected_index - 1)
        elif event.keysym == "Right":
            self.selected_index = min(max_index, self.selected_index + 1)
        elif event.keysym == "Up":
            # Bei Grid-Layout nach oben in der gleichen Spalte
            curr_idx = self.selected_index
            target_col = curr_idx % self.cols if self.cols > 0 else 0
            self.selected_index = max(0, curr_idx - self.cols)
        elif event.keysym == "Down":
            # Bei Grid-Layout nach unten in der gleichen Spalte
            curr_idx = self.selected_index
            target_col = curr_idx % self.cols if self.cols > 0 else 0
            self.selected_index = min(max_index, curr_idx + self.cols)
        elif event.keysym == "space":
            self.toggle_separator(self.selected_index)

        self.highlight_active_button()
        self.scroll_to_selected()

    def scroll_to_selected(self):
        """Scrollt zum aktuell ausgewählten Separator"""
        if not self.separator_buttons:
            return
            
        try:
            page_frame = self.separator_buttons[self.selected_index][0]
            if page_frame:
                # Berechne die Position des Frames relativ zum Canvas
                canvas_height = self.canvas.winfo_height()
                
                # Grid-Positionen sind anders als bei pack - wir müssen die Größe berechnen
                widget_info = page_frame.grid_info()
                row = widget_info['row']
                
                # Schätze die Position basierend auf Zeile und durchschnittlicher Zeilenhöhe
                avg_row_height = 250  # Geschätzte durchschnittliche Zeilenhöhe
                y_position = row * avg_row_height
                
                # Normalisiere für yview_moveto (0-1)
                container_height = self.thumbnail_container.winfo_height()
                if container_height > 0:
                    relative_pos = max(0, min(1, y_position / container_height))
                    self.canvas.yview_moveto(relative_pos)
        except Exception as e:
            print(f"Fehler beim Scrollen: {e}")

    def load_pdf(self):
        """Lädt eine PDF-Datei und zeigt Thumbnails an"""
        try:
            pdf_path = filedialog.askopenfilename(
                filetypes=[("PDF-Dateien", "*.pdf")],
                title="PDF-Datei öffnen"
            )
            
            if not pdf_path:
                return
                
            # Zeige Ladefortschritt
            self.progress.pack(fill=X, pady=5)
            self.progress.configure(value=10)
            self.root.update_idletasks()
            
            # Versuche, die PDF zu öffnen
            try:
                # Schließe vorherige Dokumente, falls vorhanden
                if self.doc:
                    self.doc.close()
                
                # Lade PDF-Dokument
                self.doc = fitz.open(pdf_path)
                self.pdf_path = pdf_path  # Nur setzen, wenn Öffnen erfolgreich
                self.progress.configure(value=30)
                self.root.update_idletasks()
            except Exception as e:
                self.progress.pack_forget()
                messagebox.showerror("Fehler", f"Die PDF-Datei konnte nicht geöffnet werden:\n{str(e)}")
                return
            
            # Pfad für Trennpunkte
            pdf_dir = os.path.dirname(self.pdf_path)
            pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
            self.saved_split_path = os.path.join(pdf_dir, f"{pdf_name}_splitpoints.json")

            # Lade gespeicherte Trennpunkte, falls vorhanden
            self.split_points.clear()
            if os.path.exists(self.saved_split_path):
                try:
                    with open(self.saved_split_path, "r") as f:
                        self.split_points = set(json.load(f))
                        self.progress.configure(value=40)
                        self.root.update_idletasks()
                except Exception as e:
                    messagebox.showerror("Fehler", f"Fehler beim Laden der Trennpunkte: {e}")
            
            # Lade die Thumbnails
            self.reload_thumbnails()
            
            # Verstecke Fortschrittsanzeige
            self.progress.pack_forget()
            
            # Aktualisiere UI-Zustand
            self.update_ui_state()
            
        except Exception as e:
            self.progress.pack_forget()
            messagebox.showerror("Fehler", f"Unerwarteter Fehler beim Laden der PDF-Datei:\n{str(e)}")

    def create_page_frame(self, parent, page_num, total_pages, show_separator=True):
        """Erstellt einen Frame für eine einzelne Seite mit Thumbnail und Trennoptionen"""
        # Äußerer Container
        page_container = ttk.Frame(parent)
        page_container.pack(fill=X, pady=10, padx=0)
        
        # Linker Marker für Trennungen
        marker_bar = ttk.Frame(page_container, width=8)
        marker_bar.pack(side=LEFT, fill=Y)
        
        # Stellen Sie sicher, dass die Breite beibehalten wird
        marker_bar.pack_propagate(False)
        
        # Haupt-Content-Frame
        content_frame = ttk.Frame(page_container)
        content_frame.pack(side=LEFT, fill=X, expand=YES, padx=100)
        
        # Seitennummer und Info
        header_frame = ttk.Frame(content_frame)
        header_frame.pack(fill=X, pady=(5, 0))
        
        page_label = ttk.Label(
            header_frame, 
            text=f"Seite {page_num + 1}", 
            font=("Segoe UI", 10, "bold"),
            bootstyle="primary"
        )
        page_label.pack(side=LEFT)
        
        # Thumbnail 
        try:
            # Erzeuge höherqualitatives Pixmap
            zoom_matrix = fitz.Matrix(self.thumbnail_scale * 2, self.thumbnail_scale * 2)
            pix = self.doc.load_page(page_num).get_pixmap(matrix=zoom_matrix, alpha=False)
            img_data = pix.samples
            
            # Konvertiere zu PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
            
            # Erstelle Tkinter-kompatibles Bild
            tk_img = ImageTk.PhotoImage(img)
            self.thumbnails.append(tk_img)  # Speichern für GC-Schutz
            
            # Container für das Bild
            img_container = ttk.Frame(content_frame, borderwidth=1, relief="solid")
            img_container.pack(pady=10, anchor=CENTER)
            
            # Bildlabel
            img_label = ttk.Label(img_container, image=tk_img)
            img_label.pack(padx=5, pady=5)
            
        except Exception as e:
            print(f"Fehler beim Laden des Thumbnails für Seite {page_num + 1}: {e}")
            error_label = ttk.Label(
                content_frame,
                text=f"Fehler beim Laden der Seite {page_num + 1}",
                bootstyle="danger"
            )
            error_label.pack(pady=20)
        
        # Separator und Button für Trennungen
        if show_separator and page_num < total_pages - 1:
            separator_frame = ttk.Frame(content_frame)
            separator_frame.pack(fill=X, pady=5)
            
            # Horizontale Trennlinie
            ttk.Separator(separator_frame, orient=HORIZONTAL).pack(fill=X, pady=5)
            
            # Trennungsbutton
            page_split = page_num + 1
            is_split_point = page_split in self.split_points
            
            # Setze Marker-Farbe basierend auf Trennungsstatus
            if is_split_point:
                marker_bar.configure(bootstyle="danger")
            else:
                marker_bar.configure(bootstyle="light")
                
            # Button zum Umschalten der Trennung
            separator_btn = ttk.Button(
                separator_frame,
                text="Trennung" if is_split_point else "Keine Trennung",
                bootstyle="danger" if is_split_point else "secondary-outline",
                command=lambda p=page_num, m=marker_bar: self.toggle_separator_at(p, m)
            )
            separator_btn.pack(fill=X, pady=5)
            
            # Speichere Referenz für Navigation
            self.separator_buttons.append((page_container, page_split, separator_btn, marker_bar))
            
        return page_container

    def reload_thumbnails(self):
        """Lädt die Thumbnails neu basierend auf dem aktuellen PDF und Zoom-Level"""
        try:
            if not self.doc or not self.pdf_path:
                return
                
            # Lösche alte Thumbnails
            for widget in self.thumbnail_container.winfo_children():
                widget.destroy()
                
            self.thumbnails.clear()
            self.separator_buttons.clear()
            
            # Zeige Fortschrittsanzeige
            self.progress.pack(fill=X, pady=5)
            self.progress.configure(value=10)
            self.root.update_idletasks()
            
            # Erstelle Thumbnails für alle Seiten
            total_pages = len(self.doc)
            
            # Berechne die Anzahl der Spalten basierend auf der Fensterbreite
            canvas_width = self.canvas.winfo_width() or 800
            # Test-Thumbnail für Größenberechnung
            try:
                test_pix = self.doc.load_page(0).get_pixmap(matrix=fitz.Matrix(self.thumbnail_scale * 2, self.thumbnail_scale * 2))
                thumbnail_width = test_pix.width + 100  # Zusätzlicher Platz für Rahmen
            except:
                thumbnail_width = 250  # Fallback-Breite
                
            # Berechne Anzahl Spalten (mind. 1, max. 7) - breiter für mehr Seiten nebeneinander
            self.cols = max(1, min(7, canvas_width // (thumbnail_width + 20)))
            
            # Grid-Layout mit roten Trennlinien zwischen den Thumbnails
            row, col = 0, 0
            
            for page_num in range(total_pages):
                # Aktualisiere Fortschrittsanzeige
                progress_value = 10 + int(80 * (page_num / total_pages))
                self.progress.configure(value=progress_value)
                self.root.update_idletasks()
                
                # Berechne Grid-Position
                grid_col = col * 2  # Doppelte Spalten für Seite + Trennlinie
                
                # Erstelle Container für diese Seite
                page_container = ttk.Frame(self.thumbnail_container)
                page_container.grid(row=row, column=grid_col, padx=20, pady=15, sticky="nsew")  # Mehr Abstand
                
                # Thumbnail erstellen
                try:
                    # Erzeuge höherqualitatives Pixmap
                    zoom_matrix = fitz.Matrix(self.thumbnail_scale * 2, self.thumbnail_scale * 2)
                    pix = self.doc.load_page(page_num).get_pixmap(matrix=zoom_matrix, alpha=False)
                    img_data = pix.samples
                    
                    # Konvertiere zu PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                    
                    # Erstelle Tkinter-kompatibles Bild
                    tk_img = ImageTk.PhotoImage(img)
                    self.thumbnails.append(tk_img)  # Speichern für GC-Schutz
                    
                    # Container für das Bild mit schwarzem Rahmen
                    img_container = ttk.Frame(page_container, borderwidth=1, relief="solid")
                    img_container.pack(fill=BOTH, expand=YES)
                    
                    # Bildlabel
                    img_label = ttk.Label(img_container, image=tk_img)
                    img_label.pack(padx=1, pady=1)
                    
                except Exception as e:
                    print(f"Fehler beim Laden des Thumbnails für Seite {page_num + 1}: {e}")
                    error_label = ttk.Label(
                        page_container,
                        text=f"Fehler bei Seite {page_num + 1}",
                        bootstyle="danger"
                    )
                    error_label.pack(pady=10)
                
                # Trennungsbereich nur hinzufügen, wenn dies nicht die letzte Seite ist

                if page_num < total_pages - 1:
                    # Erstelle separaten Frame für die Trennlinie
                    separator_frame = ttk.Frame(self.thumbnail_container, width=100)
                    separator_frame.grid(row=row, column=grid_col + 1, sticky="ns", padx=0, pady=0)
                    separator_frame.grid_propagate(False)

                    # Container für die rote Linie (NICHT für den Button)
                    line_container = ttk.Frame(separator_frame)
                    line_container.place(relx=0.5, rely=0.05, relwidth=0.15, relheight=0.9, anchor="n")

                    # Status prüfen
                    page_split = page_num + 1
                    is_split_point = page_split in self.split_points

                    # Rote Linie
                    red_line = ttk.Frame(line_container, bootstyle="danger" if is_split_point else "light")
                    red_line.place(relx=0.5, rely=0, relwidth=1.0, relheight=1.0, anchor="n")

                    # Button separat platzieren (nicht im line_container!)
                    symbol_text = "<X>" if is_split_point else "<=>"
                    bg_color = "#f0f0f0"
                    fg_color = "black"

                    symbol_label = tk.Label(
                        separator_frame,
                        text=symbol_text,
                        bg=bg_color,
                        fg=fg_color,
                        font=("Segoe UI", 10, "bold"),
                        relief="raised",
                        borderwidth=2
                    )
                symbol_label.place(relx=0.5, rely=0.5, anchor="center", width=90, height=30)

                # Klick-Event binden
                symbol_label.bind("<Button-1>", lambda e, p=page_num, r=red_line, s=symbol_label:
                                 self.toggle_separator_at(p, r, s))

                # Speichere Referenz
                self.separator_buttons.append((page_container, page_split, symbol_label, red_line))

                # Nächste Position berechnen
                col += 1
                if col >= self.cols:
                    col = 0
                    row += 1
            
            # Aktualisiere Scrollregion
            self._configure_thumbnail_area()
            
            # Verstecke Fortschrittsanzeige
            self.progress.configure(value=100)
            self.root.update_idletasks()
            self.progress.pack_forget()
            
            # Markiere aktiven Button, falls vorhanden
            if self.separator_buttons:
                self.selected_index = min(self.selected_index, len(self.separator_buttons) - 1)
                self.highlight_active_button()
            
            # Aktualisiere UI-Status
            self.update_ui_state()
            
        except Exception as e:
            self.progress.pack_forget()
            messagebox.showerror("Fehler", f"Fehler beim Laden der Thumbnails:\n{str(e)}")

    def toggle_separator_at(self, page_num, red_line, symbol_label):
        """Schaltet einen Trennpunkt an der angegebenen Seite um"""
        page_split = page_num + 1
        
        # Finde den Button-Index
        button_index = -1
        for i, (_, split, _, _) in enumerate(self.separator_buttons):
            if split == page_split:
                button_index = i
                break
                
        if button_index >= 0:
            self.toggle_separator(button_index)

    def toggle_separator(self, index):
        """Schaltet den Trennpunkt an der angegebenen Index-Position um"""
        if index < 0 or index >= len(self.separator_buttons):
            return
            
        _, page_split, symbol_label, red_line = self.separator_buttons[index]
        
        if page_split in self.split_points:
            # Entferne Trennpunkt
            self.split_points.remove(page_split)
            symbol_label.configure(text="<=>", bg="#f0f0f0", fg="black")
            red_line.configure(bootstyle="light")
        else:
            # Füge Trennpunkt hinzu
            self.split_points.add(page_split)
            symbol_label.configure(text="<X>", bg="#ff3860", fg="white")
            red_line.configure(bootstyle="danger")
        
        # Speichere automatisch
        self.save_split_points(silent=True)
        
        # Aktualisiere UI
        self.update_ui_state()

    def highlight_active_button(self):
        """Hebt den aktiv ausgewählten Separator-Label hervor"""
        if not self.separator_buttons:
            return
            
        for i, (_, _, symbol_label, _) in enumerate(self.separator_buttons):
            # Füge visuellen Fokusindikator hinzu
            if i == self.selected_index:
                # Markiere ausgewähltes Symbol mit einem Rahmen
                symbol_label.config(borderwidth=3, relief="solid")
            else:
                # Entferne Rahmen von nicht ausgewählten Symbolen
                symbol_label.config(borderwidth=2, relief="raised")


    def save_split_points(self, silent=False):
        """Speichert die Trennpunkte in eine JSON-Datei"""
        if not self.saved_split_path:
            return
            
        try:
            with open(self.saved_split_path, "w") as f:
                json.dump(sorted(list(self.split_points)), f)
                
            if not silent:
                messagebox.showinfo(
                    "Gespeichert", 
                    f"Trennpunkte wurden gespeichert in:\n{self.saved_split_path}"
                )
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern der Trennpunkte: {e}")

    def generate_split_pdfs(self):
        """Erzeugt getrennte PDF-Dateien basierend auf den Trennpunkten"""
        if not self.pdf_path:
            messagebox.showwarning("Kein PDF geladen", "Bitte laden Sie zuerst eine PDF-Datei.")
            return
            
        # Zeige Fortschrittsanzeige
        self.progress.pack(fill=X, pady=5)
        self.progress.configure(value=10)
        self.root.update_idletasks()

        pdf_dir = os.path.dirname(self.pdf_path)
        pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
        
        try:
            reader = PdfReader(self.pdf_path)
            self.progress.configure(value=20)
            self.root.update_idletasks()
            
            split_points = sorted(list(self.split_points))
            start = 0
            output_files = []
            
            # Erstelle getrennte PDFs
            total_files = len(split_points) + 1
            for i, split in enumerate(split_points):
                writer = PdfWriter()
                for page in range(start, split):
                    writer.add_page(reader.pages[page])

                output_path = os.path.join(pdf_dir, f"{pdf_name}_{str(i+1).zfill(3)}.pdf")
                output_files.append(output_path)
                
                with open(output_path, "wb") as output_pdf:
                    writer.write(output_pdf)
                
                progress = 30 + int(60 * ((i + 1) / total_files))
                self.progress.configure(value=progress)
                self.root.update_idletasks()
                
                start = split

            # Letzten Teil verarbeiten
            if start < len(reader.pages):
                writer = PdfWriter()
                for page in range(start, len(reader.pages)):
                    writer.add_page(reader.pages[page])

                output_path = os.path.join(pdf_dir, f"{pdf_name}_{str(len(split_points)+1).zfill(3)}.pdf")
                output_files.append(output_path)
                
                with open(output_path, "wb") as output_pdf:
                    writer.write(output_pdf)
            
            # Verstecke Fortschrittsanzeige
            self.progress.configure(value=100)
            self.root.update_idletasks()
            self.progress.pack_forget()
            
            # Erfolgsbenachrichtigung mit Dateipfaden
            output_msg = "\n".join(output_files)
            messagebox.showinfo(
                "Erfolg", 
                f"Getrennte PDFs wurden erfolgreich erzeugt:\n\n{output_msg}"
            )
            
        except Exception as e:
            self.progress.pack_forget()
            messagebox.showerror("Fehler", f"Fehler beim Erzeugen der PDFs: {e}")

if __name__ == "__main__":
    try:
        # Setze Error-Handler für besseres Debugging
        def show_error(exception_type, exception_value, exception_traceback):
            messagebox.showerror("Fehler", f"Unbehandelter Fehler:\n{exception_type.__name__}: {exception_value}")
            import traceback
            traceback.print_exception(exception_type, exception_value, exception_traceback)
        
        import sys
        sys.excepthook = show_error
        
        # Erstelle das Hauptfenster mit Boostrap-Stil
        root = ttk.Window(
            title="KIEBIDZ Belegsplitter",
            themename="cosmo",  # Einfacheres Theme für bessere Kompatibilität
            size=(1200, 800),
            position=(100, 100),
            minsize=(800, 600),
            resizable=(True, True)
        )
        
        # Anwendung initialisieren erst NACH dem Anzeigen des Fensters
        root.update_idletasks()  # Fenstergeometrie berechnen
        
        # Warte einen Moment, bis das Fenster sichtbar ist
        root.update()
        
        # Jetzt erst die Anwendung initialisieren
        app = PDFSplitterApp(root)
        
        # Fenster maximieren
        root.state('zoomed')
        
        # Starte Hauptschleife
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler beim Starten der Anwendung: {e}")