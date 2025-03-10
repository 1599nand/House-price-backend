const express=require("express");
const cors=require("cors");
const bodyParser = require("body-parser");


const app=express();
app.use(cors());
app.use(bodyParser.json());

app.get("/",(req,res)=> {
    res.send("Backend is running..");
});

const port=3000;

app.listen(port,()=>{
    console.log(`server is Running on ${port}`);
});