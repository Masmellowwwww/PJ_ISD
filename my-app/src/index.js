import React from 'react';
import ReactDOM from 'react-dom/client'; // ใช้ createRoot จาก React 18
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'; // เพิ่มการนำเข้า BrowserRouter, Routes, และ Route
import GetStart from './GetStart'; // นำเข้าไฟล์หน้า GetStart

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <Router> {/* ครอบด้วย Router เพื่อให้สามารถใช้ routing */}
      <Routes> {/* ใช้ Routes สำหรับการกำหนดเส้นทาง */}
        <Route path="/" element={<App />} /> {/* หน้าหลัก */}
        <Route path="/getstart" element={<GetStart />} /> {/* หน้าหลังจากคลิกปุ่ม */}
        <Route path="/app" element={<App />} /> {/* หน้าหลัก App */}
      </Routes>
    </Router>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
