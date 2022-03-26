import './App.css';
import React , {useState, useEffect} from"react";
import {BrowserRouter as Router, Route, Routes, Link} from "react-router-dom";
// import Signup from './pages/Signup'
import Login from './pages/Login'
import NavBar from './pages/NavBar'
import {useAuth, AuthProvider} from "./AuthContext";

function App() {
  // const [showSignUp, setShowSignUp] = React.useState(false);
  const [currentJoke, setCurrentJoke] = React.useState();

  const getJoke = () => {
    fetch('api/joke').then(res => res.json()).then(data => {
      setCurrentJoke(data.joke);
      console.log(data.joke)
    });
  };


  function ShowPage() {
    const {currentUser, setCurrentUser} = useAuth();
    if (currentUser) {
      return (<div>logged in
      </div>)
    }
    else {
      return <Login setCurrentUser={setCurrentUser}/>
    }
  }

  return (
    <Router>
      <AuthProvider>
        <NavBar/>
        <button onClick={getJoke}>Get Joke!</button>
        {currentJoke}
        <Routes>
          <Route path="/" element={<ShowPage/>}/>
        </Routes>
      </AuthProvider>
    </Router>
  )
}

export default App;
