//Data collection

let video;
let poseNet;
let pose;
let skeleton;

let brain;
let state="waiting";
let targetLabel;

function keyPressed(){
    if(key=="s"){
        brain.saveData();
        console.log("Data has been saved");
    }else{
        targetLabel=key;
    console.log(targetLabel);
    setTimeout(function(){
        state="collecting";
        console.log("Collection taking place");
        setTimeout(function(){
            state="waiting";
            console.log("Waiting for collection");
        }, 10000);
    }, 10000);
    } 
}

function setup(){
    createCanvas(640, 480);
    video=createCapture(VIDEO);
    video.hide();
    poseNet=ml5.poseNet(video, modelLoaded);
    poseNet.on('pose', gotPoses);

    let options={
        inputs: 34,
        outputs: 4,
        task: "classification",
        debug: true
    }

    brain=ml5.neuralNetwork(options);
    

}

function gotPoses(poses){
    // console.log(poses);
    if(poses.length>0){
        pose=poses[0].pose;
        skeleton=poses[0].skeleton;

        if(state=="collecting"){
            let inputs=[];
            for(let i=0; i<pose.keypoints.length; i++){
                let x=pose.keypoints[i].position.x;
                let y=pose.keypoints[i].position.y;

                inputs.push(x);
                inputs.push(y);
                
            }
            let target= [targetLabel];
            brain.addData(inputs, target);
        }
    }
}

function modelLoaded(){
    console.log("Model has been loaded");
}


function draw(){
    translate(video.width, 0);
    scale(-1, 1);
    image(video, 0, 0, video.width, video.height);

    if(pose){
        fill(255, 0, 0);
        for(let i=0; i<pose.keypoints.length; i++){
                fill(0, 255, 0);
                ellipse(pose.keypoints[i].position.x, pose.keypoints[i].position.y, 20);
        }
        for(let i=0;i<skeleton.length;i++){
            let a=skeleton[i][0];
            let b=skeleton[i][1];

            strokeWeight(4);
            stroke(255);
            line(a.position.x, a.position.y, b.position.x, b.position.y);
        }
    }
}