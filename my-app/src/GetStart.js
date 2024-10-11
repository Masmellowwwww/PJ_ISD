import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ReactLoading from 'react-loading';
import './App.css';
import uploadimg from './asset/image/upload.png';
import logoImage from './asset/image/Logo.png';
import searchicon from './asset/image/Search.png';
import { GoArrowLeft } from "react-icons/go";

const GetStart = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileType, setFileType] = useState('');
  const [success, setSuccess] = useState(false);
  const [plates, setPlates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  const navigate = useNavigate();

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setFileType(file.type);
  };

  const handleFileDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setFileType(file.type);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleUpload = () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const uploadUrl = fileType.startsWith('image/') 
        ? 'http://localhost:8000/upload-image/'
        : 'http://localhost:8000/upload-video/';

      setLoading(true);

      fetch(uploadUrl, {
        method: 'POST',
        body: formData
      })
        .then(response => response.json())
        .then(data => {
          console.log('Upload successful:', data);
          setPlates(data);
          setSuccess(true);
        })
        .catch(error => {
          console.error('Error uploading file:', error);
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      console.log('No file selected');
    }
  };

  const handleAppClick = () => {
    navigate('/app');
  };

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  const filteredPlates = plates.filter((item) => 
    item.plate.toLowerCase().includes(searchTerm.toLowerCase()) || 
    item.dateTime.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const goBackToUpload = () => {
    setSuccess(false);
    resetUploadState();
  };

  const resetUploadState = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setFileType('');
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

      {success ? (
        <>
          <div className='goBack'>
            <GoArrowLeft onClick={goBackToUpload} />
          </div>
          <div className='conGetStart'>
            <div className='FrameAdd' style={{ marginTop: '10px' }}>
              {previewUrl && (
                <div>
                  {fileType.startsWith('image/') ? (
                    <img 
                      src={previewUrl} 
                      alt='Uploaded Image' 
                      style={{ marginTop: '5px', maxWidth: '400px', maxHeight: '400px' }}
                    />
                  ) : fileType.startsWith('video/') ? (
                    <video 
                      controls 
                      src={previewUrl} 
                      style={{ marginTop: '5px', maxWidth: '400px', maxHeight: '400px' }}
                    >
                      Your browser does not support the video tag.
                    </video>
                  ) : (
                    <p>ไม่รองรับประเภทไฟล์นี้</p>
                  )}
                </div>
              )}
            </div>
            <div className='output'>
              <div className='Seacrh'>
                <img src={searchicon} style={{ width: "40px", height: "40px" }} />
                <input 
                  type="text" 
                  placeholder="กรอกป้ายทะเบียน" 
                  className='inputText'
                  value={searchTerm}
                  onChange={handleSearchChange}
                />
              </div>
              <div className='tablePlate'>
                <table className='detailPlate'>
                  <thead>
                    <tr>
                      <th>ป้ายทะเบียน</th>
                      <th>วัน/เวลา</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPlates.map((item, index) => (
                      <tr key={index}>
                        <td>{item.plate}</td>
                        <td>{item.dateTime}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      ) : (
        <>
          <div 
            className='FrameAdd'
            onDrop={handleFileDrop}
            onDragOver={handleDragOver}
          >
            <img src={uploadimg} alt="Upload" style={{ width: '100px' }} />
            <h1>ลากรูปภาพหรือวิดีโอมาที่นี่หรืออัปโหลดไฟล์</h1>
            <input 
              type='file' 
              accept='image/*,video/*'
              onChange={handleFileSelect}
              style={{ display: 'none' }}
              id='fileInput'
            />
            <label 
              htmlFor='fileInput'
              style={{ cursor: 'pointer', color: 'blue' }}
            >
              คลิกที่นี่เพื่ออัปโหลดไฟล์
            </label>

            {selectedFile && previewUrl && (
              <div>
                <p>ตัวอย่างไฟล์ของคุณ :</p>
                {fileType.startsWith('image/') ? (
                  <img 
                    src={previewUrl} 
                    alt='Preview' 
                    style={{ marginTop: '5px', maxWidth: '300px', maxHeight: '300px' }}
                  />
                ) : fileType.startsWith('video/') ? (
                  <video 
                    controls 
                    src={previewUrl}
                    style={{ marginTop: '5px', maxWidth: '300px', maxHeight: '300px' }}
                  >
                    ไม่รองรับประเภทไฟล์นี้
                  </video>
                ) : (
                  <p style={{ color: 'red' }}>ไม่รองรับประเภทไฟล์นี้</p>
                )}
              </div>
            )}
          </div>

          {selectedFile && previewUrl && (fileType.startsWith('image/') || fileType.startsWith('video/')) && (
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <button 
                className="btn btnupload"
                onClick={handleUpload}
                disabled={loading}
              >
                {loading ? 'กำลังอัปโหลด...' : 'อัปโหลด'}
              </button>
            </div>
          )}
        </>
      )}
	  {loading && (
  <div className="loading-overlay">
    <div className="loading-content">
      <ReactLoading type="spinningBubbles" color="#fff" height={50} width={50} />
      <p className='p-loading' style={{ color: "#fff", marginTop: '20px'}}>Loading . . .</p>
    </div>
  </div>
)}

    </div>
  );
};

export default GetStart;