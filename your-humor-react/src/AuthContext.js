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
    const [rating, setRating] = useState([0])

    useEffect(() => {
        const unsubscribe = auth.onAuthStateChanged(user => {
            setCurrentUser(user)
        })
        return unsubscribe;
    }, [])

    useEffect(() => {
        if (currentUser) {
            axios.post("https://joke-recommender-flask.herokuapp.com/add_user", {user: currentUser.email}).then((response) => {
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
        axios.post('https://joke-recommender-flask.herokuapp.com/get_joke', {user: currentUser.email}).then((response) => {
            setCurrentJoke(response.data.joke);
            setLoading(false)
            setRating([0])
        })
        .catch((err) => {
            console.log(err);
        });
    };
    
    const rateJoke = () => {
        setLoading(true)
        axios.post('https://joke-recommender-flask.herokuapp.com/rate_joke', {
            user: currentUser.email,
            joke_num: currentJoke[0],
            rating: rating[0]}).then((response) => {
                console.log(response)
                getJoke()
        })
        .catch((err) => {
            console.log(err);
        });
    };

    const signInWithGoogle = () => {
        setLoading(true);
        return signInWithPopup(auth, provider)    
    }

    const signout = () => {
        setLoading(true);
        return signOut(auth)
    }

    const logout = async () => {
        signout()
        axios.post('https://joke-recommender-flask.herokuapp.com/close_session').then((response) => {
            console.log(response);
        })
        .catch((err) => {
            console.log(err);
        });
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
        rateJoke,
        loading,
        logout
    }

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    )
}