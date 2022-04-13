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
    const [rating, setRating] = useState("Rate the joke");

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
        axios.post('api/get_joke', {user: currentUser.email}).then((response) => {
            setCurrentJoke(response.data.joke);
            setRating("Rate the joke")
        })
        .catch((err) => {
            console.log(err);
        });
        setLoading(false)
    };
    
    const rateJoke = () => {
        setLoading(true)
        axios.post('api/rate_joke', {
            user: currentUser.email,
            joke_num: currentJoke[1],
            rating: rating}).then((response) => {
                console.log(response.data)
                console.log("rated!")
            getJoke()
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
        setRating,
        rating,
        rateJoke
    }

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    )
}