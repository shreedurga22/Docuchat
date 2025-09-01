import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-file-reader',
  templateUrl: './file-reader.component.html',
  styleUrls: ['./file-reader.component.css']
})
export class FileReaderComponent {
  selectedFile: File | null = null;
  uploadedText: string = '';
  response: string = '';
  question: string = '';
  showModal: boolean = false;

  private apiUrl = 'http://localhost:5000'; // Make sure this matches your Flask backend

  constructor(private http: HttpClient) {}

  // Handle file selection
  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.uploadedText = '';
      this.response = '';
      this.question = '';
      this.showModal = false;
    }
  }

  // Remove selected file
  removeFile() {
    this.selectedFile = null;
    this.uploadedText = '';
    this.response = '';
    this.question = '';
    this.showModal = false;
  }

  // Upload the selected file
  onUpload() {
    if (!this.selectedFile) {
      alert('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    // DO NOT manually set Content-Type; Angular will handle it
    this.http.post<{ text_preview?: string }>(`${this.apiUrl}/upload`, formData).subscribe({
      next: (res) => {
        this.uploadedText = res.text_preview || '';
        this.showModal = true;
      },
      error: (err: any) => {
        console.error('Upload error:', err);
        alert('Failed to upload file.');
      }
    });
  }

  // Ask a question about the uploaded file
  onAsk() {
    if (!this.question.trim() || !this.uploadedText) {
      alert('Please enter a question and upload a file first.');
      return;
    }

    this.http.post<{ answer?: string }>(`${this.apiUrl}/ask`, {
      question: this.question
    }).subscribe({
      next: (res) => {
        this.response = res.answer || 'No answer returned.';
      },
      error: (err: any) => {
        console.error('Backend error:', err);
        this.response = 'Could not get answer from backend.';
      }
    });
  }

  // Close modal
  onModalOk() {
    this.showModal = false;
  }

  // Strip markdown and remove short heading
  stripMarkdown(text: string): string {
    if (!text) return '';
    text = text.replace(/\*\*(.*?)\*\*/g, '$1')
               .replace(/\*(.*?)\*/g, '$1')
               .replace(/__(.*?)__/g, '$1')
               .replace(/_(.*?)_/g, '$1');

    const lines = text.trim().split('\n');
    if (lines.length > 1 && lines[0].length < 40) {
      return lines.slice(1).join('\n').trim();
    }
    return text.trim();
  }
}
