import React from 'react';
import {useAuth} from "../AuthContext";

function Login() {
  const { signInWithGoogle } = useAuth();

  return (
    <div>
        Login
        <button className="login-with-google-btn" onClick={signInWithGoogle}>Sign in with Google</button>
    </div>
  )
}

export default Login;