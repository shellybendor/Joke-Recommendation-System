import React, {useContext, useState, useEffect} from 'react'
import {auth, db} from "../firebase-config"
import firebase from 'firebase';

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

    const value = {
        currentUser,
    }
    
    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    )
}