const input=document.getElementById("chatInput");

const messages=document.getElementById("chatMessages");

document.getElementById("sendChat").onclick=sendMessage;

async function sendMessage(){

    const question=input.value.trim();

    if(!question)return;

    messages.innerHTML+=`

    <div class="user">

        ${question}

    </div>

    `;

    input.value="";

    const response=await fetch("/chatbot/",{

        method:"POST",

        headers:{
            "Content-Type":"application/json"
        },

        body:JSON.stringify({

            question

        })

    });

    const data=await response.json();

    messages.innerHTML+=`

    <div class="bot">

        ${data.answer}

    </div>

    `;

    messages.scrollTop=messages.scrollHeight;

}