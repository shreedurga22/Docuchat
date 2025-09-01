import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { FileReaderComponent } from './file-reader/file-reader.component';

const routes: Routes = [
  { path: '', component: FileReaderComponent },
  { path: '**', redirectTo: '' }  // Optional: redirect unknown routes to home
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
