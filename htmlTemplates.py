css = '''
<style>
.chat-message {	
    padding: 0.1rem; border-radius: 0.1rem; margin-bottom: 0.5rem; display: flex;
}
.chat-message.user {
    background-color: #63e0a6
}
.chat-message.bot {
    background-color: #ccd3e0
}
.chat-message .avatar {
    width: 10%;
}
.chat-message .avatar img {
    max-width: 58px;
    max-height: 58px;
    border-radius: 90%;
    object-fit: cover;
}
.chat-message .message {
    width: 85%;
    padding: 1.0rem;
    color: #000000;
}

'''

bot_template = '''
    <div class="chat-message bot">
        <div class="avatar">
            <img src="https://cdn-icons-png.flaticon.com/512/4616/4616271.png">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
'''

user_template = '''
    <div class="chat-message user">
        <div class="avatar">
            <img src="https://static.vecteezy.com/system/resources/previews/018/742/015/original/minimal-profile-account-symbol-user-interface-theme-3d-icon-rendering-illustration-isolated-in-transparent-background-png.png">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
'''