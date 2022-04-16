// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// import {getFirestore} from 'firebase/firestore';
import {getAuth, GoogleAuthProvider} from 'firebase/auth';
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAWY5RA-939_94IKYB0fs9JgJg_DA313Cg",
  authDomain: "your-humor.firebaseapp.com",
  projectId: "your-humor",
  storageBucket: "your-humor.appspot.com",
  messagingSenderId: "638737699387",
  appId: "1:638737699387:web:cf1bae9ad2aeff3b4b021e"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// export const db = getFirestore(app);
export const auth = getAuth(app);
export const provider = new GoogleAuthProvider();