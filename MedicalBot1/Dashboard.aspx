<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Dashboard.aspx.cs" Inherits="MedicalBotWeb.Dashboard" Async="true" %>

<!DOCTYPE html>
<html lang="es">
<head runat="server">
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>MedicalBot - Plataforma</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>
    /* Variables y estilos base */
    :root {
      --primary-color: #1a73e8;
      --primary-dark: #0d47a1;
      --secondary-color: #34a853;
      --danger-color: #ea4335;
      --warning-color: #fbbc05;
      --light-color: #f8f9fa;
      --dark-color: #202124;
      --gray-color: #5f6368;
      --light-gray: #e8eaed;
      --white: #ffffff;
      --shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
      --border-radius: 8px;
      --transition: all 0.3s ease;
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', sans-serif;
      background-color: #f5f7fa;
      color: var(--dark-color);
      line-height: 1.6;
      display: flex;
      min-height: 100vh;
    }

    form {
      width: 100%;
    }

    .sidebar {
      width: 250px;
      background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
      color: var(--white);
      padding: 20px 0;
      height: 100vh;
      position: fixed;
      box-shadow: var(--shadow);
      z-index: 100;
    }

    .sidebar h2 {
      text-align: center;
      margin-bottom: 30px;
      font-size: 1.5rem;
      font-weight: 600;
      padding: 0 20px;
    }

    .sidebar ul {
      list-style: none;
    }

    .sidebar li a {
      display: block;
      padding: 12px 20px;
      color: var(--white);
      text-decoration: none;
      transition: var(--transition);
      border-left: 4px solid transparent;
      font-size: 0.95rem;
    }

    .sidebar li a:hover {
      background-color: rgba(255, 255, 255, 0.1);
      border-left: 4px solid var(--white);
    }

    .content {
      margin-left: 250px;
      width: calc(100% - 250px);
      padding: 20px;
      background-color: var(--white);
      min-height: 100vh;
    }

    .section {
      background-color: var(--white);
      border-radius: var(--border-radius);
      box-shadow: var(--shadow);
      padding: 25px;
      margin-bottom: 20px;
      display: none;
    }

    .section.active {
      display: block;
    }

    .section-title {
      font-size: 1.5rem;
      font-weight: 600;
      color: var(--primary-color);
      margin-bottom: 20px;
      padding-bottom: 10px;
      border-bottom: 2px solid var(--light-gray);
    }

    .section-content {
      line-height: 1.8;
    }

    .section-content p {
      margin-bottom: 15px;
    }

    .section-content ul {
      margin: 15px 0 15px 20px;
    }

    .section-content li {
      margin-bottom: 8px;
    }

    .disclaimer {
      background-color: #f8f9fa;
      padding: 15px;
      border-left: 4px solid var(--warning-color);
      border-radius: 4px;
      font-size: 0.9rem;
    }

    .chat-container {
      display: flex;
      flex-direction: column;
      height: calc(100vh - 40px);
      max-height: 800px;
      background-color: var(--white);
      border-radius: var(--border-radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .chat-header {
      background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
      color: var(--white);
      padding: 15px 20px;
      display: flex;
      align-items: center;
      gap: 15px;
    }

    .bot-avatar {
      width: 40px;
      height: 40px;
      background-color: var(--white);
      color: var(--primary-color);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: bold;
      font-size: 1.1rem;
    }

    .chat-body {
      flex-grow: 1;
      padding: 20px;
      overflow-y: auto;
      background-color: #f9f9f9;
      border-bottom: 1px solid var(--light-gray);
    }

    .chat-message {
      margin-bottom: 15px;
      max-width: 80%;
      animation: fadeIn 0.3s ease-out;
    }

    .user-message {
      margin-left: auto;
      background-color: var(--primary-color);
      color: var(--white);
      padding: 10px 15px;
      border-radius: 18px 18px 0 18px;
    }

    .bot-message {
      margin-right: auto;
      background-color: var(--white);
      color: var(--dark-color);
      padding: 10px 15px;
      border-radius: 18px 18px 18px 0;
      border: 1px solid var(--light-gray);
      box-shadow: var(--shadow);
    }

    .chat-input {
      padding: 15px;
      background-color: var(--white);
    }

    .input-chat {
      width: 100%;
      padding: 12px 15px;
      border: 1px solid var(--light-gray);
      border-radius: var(--border-radius);
      font-size: 0.95rem;
      transition: var(--transition);
    }

    .input-chat:focus {
      outline: none;
      border-color: var(--primary-color);
      box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2);
    }

    .send-button, .refresh-button, .report-button {
      padding: 10px 20px;
      border: none;
      border-radius: var(--border-radius);
      font-weight: 500;
      cursor: pointer;
      transition: var(--transition);
    }

    .send-button {
      background-color: var(--primary-color);
      color: var(--white);
    }

    .refresh-button {
      background-color: var(--light-gray);
      color: var(--dark-color);
    }

    .report-button {
      background-color: var(--danger-color);
      color: var(--white);
    }

    .doctor-image-container {
      text-align: center;
      margin: 20px 0;
    }

    .doctor-image {
      max-width: 200px;
      border-radius: 50%;
      border: 5px solid var(--primary-color);
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .welcome-message {
      text-align: center;
      margin-bottom: 20px;
      font-size: 1.1rem;
      color: var(--dark-color);
    }

    @media (max-width: 768px) {
      .sidebar {
        width: 100%;
        height: auto;
        position: relative;
      }

      .content {
        margin-left: 0;
        width: 100%;
      }

      .chat-container {
        height: auto;
        max-height: none;
      }
    }

    .hidden {
      display: none !important;
    }

    .emergency-number {
      font-size: 1.8rem;
      font-weight: bold;
      color: var(--danger-color);
      margin: 20px 0;
      text-align: center;
    }

    .emergency-contact {
      display: flex;
      justify-content: space-between;
      max-width: 400px;
      margin: 0 auto;
      padding: 10px 0;
    }

    .emergency-contact p {
      font-weight: 500;
    }
  </style>
</head>
<body>
  <form id="form1" runat="server">
    <asp:ScriptManager ID="ScriptManager1" runat="server"></asp:ScriptManager>

    <div class="sidebar">
      <h2>MedicalBot</h2>
      <ul>
        <li><a href="#" onclick="mostrarSeccion('quienes')">🧑‍⚕️ Quienes somos</a></li>
        <li><a href="#" onclick="mostrarSeccion('urgente')">🚨 Atención urgente</a></li>
        <li><a href="#" onclick="mostrarSeccion('chatbot')">💬 ChatBot</a></li>
        <li><a href="#" onclick="mostrarSeccion('cita')">🗓️ Agendar cita</a></li>
        <li><a href="#" onclick="mostrarSeccion('emergencia')">📞 Números de emergencia</a></li>
      </ul>
    </div>

    <div class="content">
      <asp:UpdatePanel ID="UpdatePanelChat" runat="server">
        <ContentTemplate>
          <div id="chatbot" class="section active">
            <div class="chat-container">
              <div class="chat-header">
                <div class="bot-avatar">MB</div>
                <span>MedicalBot - Tu asistente médico virtual</span>
                <div style="display: flex; gap: 10px;">
                  <asp:Button ID="btnNuevoChat" runat="server" CssClass="refresh-button" Text="Nuevo" OnClick="btnNuevoChat_Click" />
                  <asp:Button ID="btnReporte" runat="server" CssClass="report-button" Text="Reporte" OnClick="btnReporte_Click" />
                </div>
              </div>
              <div class="chat-body" id="chatBody">
                <div class="bot-message">Hola, soy MedicalBot. Por favor, completa todos los campos para recibir una orientación.</div>
                <div class="doctor-image-container">
                  <img src="cuybot.jpeg" alt="Doctor" class="doctor-image" />
                  <div class="welcome-message">Estoy aquí para ayudarte con tus consultas médicas</div>
                </div>
                <asp:Literal ID="litHistorial" runat="server" />
              </div>
              <div class="chat-input" style="display: flex; flex-direction: column; gap: 10px;">
                <asp:TextBox ID="txtNombre" runat="server" CssClass="input-chat" placeholder="Tu nombre completo:" />
                <asp:TextBox ID="txtEdad" runat="server" CssClass="input-chat" placeholder="Tu edad:" TextMode="Number" />
                <asp:TextBox ID="txtActividad" runat="server" CssClass="input-chat" placeholder="¿Qué estabas haciendo antes de los síntomas?" />
                <asp:TextBox ID="txtSintoma" runat="server" CssClass="input-chat" placeholder="Describe tus síntomas aquí..." />
                <asp:TextBox ID="txtAlergía" runat="server" CssClass="input-chat" placeholder="¿Tienes alguna alergía?" />
                <asp:Button ID="btnEnviar" runat="server" Text="Enviar" CssClass="send-button" />
              </div>
            </div>
          </div>
        </ContentTemplate>
      </asp:UpdatePanel>

      <div id="quienes" class="section hidden">
        <div class="section-title">Quienes somos</div>
        <div class="section-content">
          <p><strong>MedicalBot</strong> es un asistente médico virtual de vanguardia, impulsado por inteligencia artificial...</p>
          <p class="disclaimer"><strong>Importante:</strong> No sustituye el diagnóstico profesional.</p>
        </div>
      </div>

      <div id="urgente" class="section hidden">
        <div class="section-title">Atención Urgente</div>
        <div class="section-content">
          <p>Busca ayuda inmediata si tienes alguno de estos síntomas graves:</p>
          <ul>
            <li>Dificultad para respirar</li>
            <li>Dolor en el pecho</li>
            <li>Pérdida de conciencia</li>
          </ul>
        </div>
      </div>

<div id="cita" class="section hidden">
  <div class="section-title">Agendar Cita</div>
  <div class="section-content">
    <p>Selecciona una fecha y hora para tu cita médica:</p>
    <div style="display: flex; flex-direction: column; gap: 15px; max-width: 400px; margin: 20px auto;">
      <label for="nombrePaciente">Tu nombre completo:</label>
      <input type="text" id="nombrePaciente" class="input-chat" placeholder="Ej. Juan Pérez">
      
      <label for="fechaCita">Selecciona una fecha:</label>
      <input type="date" id="fechaCita" class="input-chat">
      
      <label for="horaCita">Selecciona una hora:</label>
      <select id="horaCita" class="input-chat">
        <option value="">-- Selecciona un horario --</option>
        <option value="08:00">08:00 AM</option>
        <option value="09:00">09:00 AM</option>
        <option value="10:00">10:00 AM</option>
        <option value="11:00">11:00 AM</option>
        <option value="14:00">02:00 PM</option>
        <option value="15:00">03:00 PM</option>
        <option value="16:00">04:00 PM</option>
      </select>

      <!-- Botón que NO recarga página -->
      <button type="button" onclick="confirmarCita()" class="send-button">Agendar cita</button>

      <!-- Confirmación de cita -->
      <div id="mensajeConfirmacion" style="margin-top: 15px; font-weight: bold;"></div>

      <!-- Historial de citas -->
      <div id="historialCitas" style="margin-top: 20px;">
        <h4 style="text-align: center; color: var(--primary-color); margin-bottom: 10px;">Historial de citas</h4>
        <ul id="listaCitas" style="list-style: none; padding-left: 0;"></ul>
      </div>
    </div>
  </div>
</div>


<div id="emergencia" class="section hidden">
  <div class="section-title">📞 Números de Emergencia</div>
  <div class="section-content">
    
    <p style="text-align: center; font-style: italic; margin-bottom: 20px; color: var(--gray-color);">
      "En los momentos difíciles, pedir ayuda es un acto de valentía."
    </p>

    <div style="display: flex; flex-direction: column; gap: 20px;">

      <div style="display: flex; align-items: center; background-color: #ffe5e5; border-left: 6px solid #d32f2f; padding: 15px; border-radius: 10px;">
        <span style="font-size: 2rem; margin-right: 15px;">🚑</span>
        <div>
          <strong style="font-size: 1.2rem;">Emergencias Generales:</strong>
          <div style="font-size: 1.5rem; font-weight: bold;">911</div>
          <small style="color: #a30000;">Llama sin dudar si tu vida o la de alguien más está en riesgo.</small>
        </div>
      </div>

      <div style="display: flex; align-items: center; background-color: #fff3e0; border-left: 6px solid #f57c00; padding: 15px; border-radius: 10px;">
        <span style="font-size: 2rem; margin-right: 15px;">🔥</span>
        <div>
          <strong style="font-size: 1.2rem;">Bomberos:</strong>
          <div style="font-size: 1.3rem; font-weight: bold;">119</div>
          <small style="color: #a15b00;">Actúan con rapidez ante incendios y emergencias.</small>
        </div>
      </div>

      <div style="display: flex; align-items: center; background-color: #e3f2fd; border-left: 6px solid #1976d2; padding: 15px; border-radius: 10px;">
        <span style="font-size: 2rem; margin-right: 15px;">👮‍♂️</span>
        <div>
          <strong style="font-size: 1.2rem;">Policía Nacional:</strong>
          <div style="font-size: 1.3rem; font-weight: bold;">112</div>
          <small style="color: #0d47a1;">Protegiéndote en situaciones de peligro o violencia.</small>
        </div>
      </div>

      <div style="display: flex; align-items: center; background-color: #f3e5f5; border-left: 6px solid #8e24aa; padding: 15px; border-radius: 10px;">
        <span style="font-size: 2rem; margin-right: 15px;">🛡️</span>
        <div>
          <strong style="font-size: 1.2rem;">Defensa Civil:</strong>
          <div style="font-size: 1.3rem; font-weight: bold;">144</div>
          <small style="color: #5e0079;">Apoyo ante desastres naturales o evacuaciones.</small>
        </div>
      </div>

    </div>

    <p style="text-align: center; margin-top: 30px; font-style: italic; color: #4a4a4a;">
      "Recuerda, no estás solo. Hay ayuda disponible las 24 horas, los 7 días de la semana."
    </p>
  </div>
</div>
a
    </div>
  </form>

  <script>
      function mostrarSeccion(seccionId) {
          document.querySelectorAll('.section').forEach(section => {
              section.classList.add('hidden');
              section.classList.remove('active');
          });
          const seccion = document.getElementById(seccionId);
          if (seccion) {
              seccion.classList.remove('hidden');
              seccion.classList.add('active');
          }
          document.querySelector('.content').scrollTo(0, 0);
      }

      function confirmarCita() {
          const nombre = document.getElementById('nombrePaciente').value;
          const fecha = document.getElementById('fechaCita').value;
          const hora = document.getElementById('horaCita').value;
          const mensaje = document.getElementById('mensajeConfirmacion');

          if (!nombre || !fecha || !hora) {
              mensaje.style.color = "red";
              mensaje.textContent = "Por favor completa todos los campos para agendar la cita.";
              return;
          }

          // Obtener el día de la semana
          const dias = ['domingo', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado'];
          const fechaObj = new Date(fecha);
          const diaSemana = dias[fechaObj.getDay()];

          // Formatear mensaje
          mensaje.style.color = "green";
          mensaje.innerHTML = `✅ ¡Cita agendada! <strong>${nombre}</strong>, te esperamos el <strong>${diaSemana} ${fecha}</strong> a las <strong>${hora}</strong>.`;
      }
  </script>
</body>
</html>
