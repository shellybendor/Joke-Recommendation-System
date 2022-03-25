import './App.css';
import React , {useState} from "react";
import {BrowserRouter as Router, Switch, Route, Routes, Link} from "react-router-dom";
import Signup from './pages/Signup'
import Login from './pages/Login'
import {useAuth, AuthProvider} from "./AuthContext";

function App() {
  const [showSignUp, setShowSignUp] = React.useState(false);

  function ShowPage() {
    const {currentUser} = useAuth();
    if (showSignUp) {
      return <Signup/>
    }
    else {
      return <Login/>
    }
  }

  return (
    <Router>
      <AuthProvider>
        <Routes>
          <Route path="/">
            <ShowPage />
          </Route>
        </Routes>
      </AuthProvider>
    </Router>
  )
}

export default App;
