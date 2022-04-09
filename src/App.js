import './App.css';
import React from "react";
import LoginCard from './components/LoginCard'
import JokeCard from './components/JokeCard'
import {useAuth, AuthProvider} from "./AuthContext";

function App() {
  
  function ShowPage() {
    const {currentUser} = useAuth();
    if (currentUser) {
      return (
      <JokeCard/>)
    }
    else {
      return <LoginCard/>
    }
  }

  return (
      <AuthProvider>
        <ShowPage/>
      </AuthProvider>
  )
}

export default App;
