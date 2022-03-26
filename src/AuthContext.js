import axios from 'axios';
import React, {useContext, useState, useEffect} from 'react'
import {auth, db, provider} from "./firebase-config"
// import firebase from 'firebase';
import {signInWithPopup, signOut} from 'firebase/auth';
// import { useNavigate } from 'react-router-dom';

/**
 * Authentication context for user signup with firebase
 */

 const AuthContext = React.createContext();

 export function useAuth() {
     return useContext(AuthContext)
 }

export function AuthProvider({children}) {
    const [currentUser, setCurrentUser] = useState();
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (currentUser) {
            axios.post("/add_user", {"user": currentUser}).then((response) => {
                console.log(response);
            })
            .catch((err) => {
                console.log(err);
              });
        }
    }, [currentUser])

    const signInWithGoogle = () => {
        signInWithPopup(auth, provider).then((result) => {
            localStorage.setItem("currentUser", result.user.uid);
            setCurrentUser(result.user.email)
          
          })
    
    }

    const signout = () => {
        signOut(auth).then(() => {
            localStorage.clear();
            setCurrentUser();
        })
    }

    const value = {
        currentUser,
        setCurrentUser,
        signInWithGoogle,
        signout,
    }

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    )
}