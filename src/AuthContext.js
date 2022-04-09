import axios from 'axios';
import React, {useContext, useState, useEffect} from 'react'
import {auth, provider} from "./firebase-config"
import {signInWithPopup, signOut} from 'firebase/auth';

/**
 * Authentication context for user signup with firebase
 */

 const AuthContext = React.createContext();

 export function useAuth() {
     return useContext(AuthContext)
 }

export function AuthProvider({children}) {
    const [currentUser, setCurrentUser] = useState("");
    const [currentJoke, setCurrentJoke] = useState([null, 'Please wait while we get your joke']);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const unsubscribe = auth.onAuthStateChanged(user => {
            setCurrentUser(user)
        })
        return unsubscribe;
    }, [])

    useEffect(() => {
        if (currentUser) {
            axios.post("/api/add_user", {user: currentUser.email}).then((response) => {
                console.log(response);
                getJoke();
            })
            .catch((err) => {
                console.log(err);
            });
        }
        setLoading(false)
    }, [currentUser])

    const getJoke = () => {
        setLoading(true)
        axios.post('api/joke', {user: currentUser.email}).then((response) => {
          setCurrentJoke(response.data.joke);
        })
        .catch((err) => {
            console.log(err);
        });
        setLoading(false)
      };

    const signInWithGoogle = () => {
        setLoading(true);
        return signInWithPopup(auth, provider)    
    }

    const signout = () => {
        setLoading(true);
        return signOut(auth)
    }

    const value = {
        currentUser,
        setCurrentUser,
        signInWithGoogle,
        signout,
        getJoke,
        currentJoke,
    }

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    )
}