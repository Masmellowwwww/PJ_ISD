import React from 'react';
import { useNavigate } from 'react-router-dom'; // เพิ่มการนำเข้า useNavigate
import './App.css';
import carImage from './asset/image/car.png';
import AddPhoto from './asset/image/Add_photo.png';
import AddVideo from './asset/image/Add_video.png';
import logoImage from './asset/image/Logo.png';

const App = () => {
  const navigate = useNavigate(); // สร้างฟังก์ชัน navigate สำหรับการนำทาง

  const handleGetStartClick = () => {
    navigate('/getstart'); // นำทางไปยังหน้า GetStart.js
  };
  const handleAppClick = () => {
    navigate('/app'); 
  };

  return (
    
    <div className="container">
       <nav className="navbar">
      <img 
    src={logoImage}
    alt="PlateX Logo" 
    className="logo" 
    onClick={handleAppClick} 
    style={{ cursor: 'pointer' }} 
  />
      </nav>

      <div className="slogan">
        <div className="Text">
          <h1 style={{fontSize: '40px'}}>license plate recognition</h1>
          <p>
          License Plate Recognition (LPR) หรือการรู้จำป้ายทะเบียน เป็นเทคโนโลยีที่ใช้ในการตรวจจับและแยกแยะตัวอักษรและตัวเลขจากป้ายทะเบียนยานพาหนะ โดยอาศัยภาพที่ได้รับจากกล้องถ่ายรูปหรือวิดีโอ
          </p>
          <button className="btn" onClick={handleGetStartClick}>เริ่มต้นใช้งาน</button> {/* ใช้ฟังก์ชัน handleGetStartClick */}
        </div>
        <div className="image-container">
          <img src={carImage} alt="Car" className="image" />
        </div>
      </div>

      <div className="feature">
        <h1 className='H1F'>Feature</h1>
        <div className="detailfeature">
          <div className="AddPhoto">
            <img src={AddPhoto} alt="Add Photo" />
            <h3>Upload Photo</h3>
            <p style={{ color: '#404040' }}>
              สามารถอัปโหลดรูป เพื่อตรวจจับป้ายทะเบียนรถ
            </p>
          </div>
          <div className="AddVideo">
            <img src={AddVideo} alt="Add Video" />
            <h3>Upload Video</h3>
            <p style={{ color: '#404040' }}>
              สามารถอัปโหลดวิดีโอ เพื่อตรวจจับป้ายทะเบียนรถ
            </p>
          </div>
        </div>
      </div>

      <footer className="footer">
        <p>© 2024 PlateX. All rights reserved.</p>
        <p>Create Project by 65070172 65070234</p>
      </footer>
    </div>
  );
};

export default App;
