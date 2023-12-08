//Final detection

let video;
let poseNet;
let pose;
let skeleton;

let brain;
let state="waiting";
let targetLabel;

let poseLabel= "standing";

// function keyPressed(){
//     if(key=="s"){
//         brain.saveData();
//     }else{
//         targetLabel=key;
//     console.log(targetLabel);
//     setTimeout(function(){
//         state="collecting";
//         console.log("Collection taking place");
//         setTimeout(function(){
//             state="waiting";
//             console.log("Waiting for collection");
//         }, 10000);
//     }, 10000);
//     } 
// }

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

    const modelInfo = {
        model: 'model\\model.json',
        metadata: 'model\\model_meta.json',
        weights: 'model\\model.weights.bin'
    };
    brain.load(modelInfo, brainLoaded);
}

function brainLoaded(){
    console.log("brain loaded");

    classifyPose();
}

function classifyPose(){
    if(pose){
        let inputs=[];
        for(let i=0; i<pose.keypoints.length; i++){
            let x=pose.keypoints[i].position.x;
            let y=pose.keypoints[i].position.y;
            inputs.push(x);
            inputs.push(y);
        }

        brain.classify(inputs, gotResults);
    }else{
        setTimeout(classifyPose, 100);
    }
}



function gotResults(error, results){
    
        if(results[0].label==0) poseLabel="Standing";
        else if(results[0].label==1) poseLabel="Vrikshasana";
        else if(results[0].label==2) poseLabel="Sukhasana";
        else if(results[0].label==3) poseLabel="Utkatasana";
        else if(results[0].label==4) poseLabel="Vajrasana";
        else if(results[0].label==5) poseLabel="Ado mukha swanasana";
        else if(results[0].label==6) poseLabel="Veerabhadrasana";
        else if(results[0].label==7) poseLabel="Anjaneyasana";

        if(results[0].confidence<=0.7){
            poseLabel="...";
        }


    console.log(results[0].confidence);
    classifyPose();
}
// function dataReady(){
//     brain.normalizeData();
//     brain.train({epochs: 50}, finished);
// }

// function finished(){
//     console.log('Model trained');
//     brain.save();
// }

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
    push();
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
    pop();
    fill(1, 1, 1);
    stroke(255);
    strokeWeight(4);
    textSize(50);
    textAlign(CENTER, BOTTOM);
    text(poseLabel, width/2, height/2);
}