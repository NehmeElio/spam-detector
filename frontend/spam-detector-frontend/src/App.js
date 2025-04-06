import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);

  const handleInputChange = (event) => {
    setText(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setPrediction('');
    
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL
      const response = await axios.post('${backendUrl}:3000/predict', {
        text: text,
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("There was an error!", error);
      setPrediction('Error in prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Spam or Ham Detector</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={handleInputChange}
          placeholder="Enter text to classify"
          rows="6"
          cols="50"
        />
        <br />
        <button type="submit" disabled={loading}>
          {loading ? 'Loading...' : 'Submit'}
        </button>
      </form>
      {prediction && <h3>Prediction: {prediction}</h3>}
    </div>
  );
}

export default App;
