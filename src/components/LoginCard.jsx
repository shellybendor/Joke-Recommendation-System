import React from 'react';
import {useAuth} from "../AuthContext";

export default function LoginCard() {
  const { signInWithGoogle } = useAuth();

  return (

    <div className="box">
            <h2>Welcome to 'Your Humor'!</h2>
            <p>After rating 10 jokes, I will recommend jokes based on your humor <span role='img'>🤣</span></p>
            <p>Please login to get recommended jokes.</p>
            <button className="box-button" onClick={signInWithGoogle}>Sign in with Google</button>
        </div>
  )
}