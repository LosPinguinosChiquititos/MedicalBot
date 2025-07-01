using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Web.UI;
using Newtonsoft.Json;

namespace MedicalBotWeb
{
    public class PredictionRequest
    {
        public string patientName { get; set; }
        public int patientAge { get; set; }
        public string symptoms { get; set; }
        public string previousActivities { get; set; }
        public string allergies { get; set; } // Este campo es clave
    }

    public class PredictionResponse
    {
        public string predicted_disease { get; set; }
        public string summary { get; set; }
        public string treatment { get; set; }
        public string error { get; set; }
    }

    public partial class Dashboard : System.Web.UI.Page
    {
        private static readonly HttpClient client = new HttpClient();
        private readonly string api_url = "http://127.0.0.1:5000/predict";

        protected void Page_Load(object sender, EventArgs e)
        {
            if (!IsPostBack)
            {
                litHistorial.Text = "<div class='message bot-message'>Hola, soy MedicalBot. Por favor, completa todos los campos para recibir una orientación.</div>";
            }
            ScriptManager.GetCurrent(this).RegisterAsyncPostBackControl(btnEnviar);
            btnEnviar.Click += new EventHandler(btnEnviar_Click_Async);
        }

        protected async void btnEnviar_Click_Async(object sender, EventArgs e)
        {
            string nombre = txtNombre.Text.Trim();
            string actividad = txtActividad.Text.Trim();
            string sintoma = txtSintoma.Text.Trim();
            string alergia = txtAlergía.Text.Trim(); // CAMBIO: Leemos el nuevo campo

            if (string.IsNullOrWhiteSpace(nombre) || string.IsNullOrWhiteSpace(sintoma) || string.IsNullOrWhiteSpace(txtEdad.Text))
            {
                litHistorial.Text += "<div class='message bot-message error'>Por favor, completa los campos de nombre, edad y síntomas.</div>";
                return;
            }

            if (!int.TryParse(txtEdad.Text, out int edad))
            {
                litHistorial.Text += "<div class='message bot-message error'>La edad debe ser un número válido.</div>";
                return;
            }

            // CAMBIO: Mostramos también la alergia en el mensaje del usuario
            string userMessageHtml = $@"
                <div class='message user-message'>
                    <p><strong>Paciente:</strong> {nombre}, {edad} años</p>
                    <p><strong>Síntomas:</strong> {sintoma}</p>
                    <p><strong>Actividad Previa:</strong> {actividad}</p>
                    <p><strong>Alergias conocidas:</strong> {(string.IsNullOrWhiteSpace(alergia) ? "Ninguna" : alergia)}</p>
                </div>";
            litHistorial.Text += userMessageHtml;

            txtSintoma.Text = "";
            txtAlergía.Text = ""; // Limpiamos el campo de alergia también

            litHistorial.Text += "<div class='message bot-message'><i>Analizando, por favor espera...</i></div>";

            try
            {
                // CAMBIO: Incluimos la alergia en el payload de la petición
                var requestPayload = new PredictionRequest
                {
                    patientName = nombre,
                    patientAge = edad,
                    symptoms = sintoma,
                    previousActivities = actividad,
                    allergies = alergia
                };

                var jsonPayload = JsonConvert.SerializeObject(requestPayload);
                var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

                HttpResponseMessage response = await client.PostAsync(api_url, content);
                string jsonResponse = await response.Content.ReadAsStringAsync();

                string botResponseHtml;
                if (response.IsSuccessStatusCode)
                {
                    var prediction = JsonConvert.DeserializeObject<PredictionResponse>(jsonResponse);
                    if (prediction != null && string.IsNullOrEmpty(prediction.error))
                    {
                        // El tratamiento que viene de la API ya puede incluir la advertencia
                        botResponseHtml = $@"
                            <div class='message bot-message'>
                                <p>Basado en el análisis, la posible condición es: <strong>{prediction.predicted_disease}</strong></p>
                                <p><strong>Resumen:</strong> {prediction.summary}</p>
                                <p><strong>Tratamiento Sugerido:</strong> {prediction.treatment}</p> 
                                <p class='disclaimer'><strong>Advertencia:</strong> Este es un diagnóstico preliminar y no reemplaza la consulta con un profesional médico.</p>
                            </div>";
                    }
                    else
                    {
                        botResponseHtml = $"<div class='message bot-message error'>Error en la API: {prediction?.error ?? jsonResponse}</div>";
                    }
                }
                else
                {
                    botResponseHtml = $"<div class='message bot-message error'>No se pudo conectar con el servicio de IA (Código: {response.StatusCode}). Detalles: {jsonResponse}</div>";
                }
                litHistorial.Text += botResponseHtml;
            }
            catch (Exception ex)
            {
                litHistorial.Text += $"<div class='message bot-message error'>Error de conexión: {ex.Message}. Asegúrate de que el servidor de Python esté en ejecución.</div>";
            }
        }

        protected void btnNuevoChat_Click(object sender, EventArgs e)
        {
            litHistorial.Text = "<div class='message bot-message'>Hola, soy MedicalBot. Por favor, completa todos los campos para recibir una orientación.</div>";
            txtNombre.Text = "";
            txtEdad.Text = "";
            txtActividad.Text = "";
            txtSintoma.Text = "";
            txtAlergía.Text = "";
        }

        protected void btnReporte_Click(object sender, EventArgs e)
        {
            litHistorial.Text += "<div class='message bot-message'>[Función de reporte en desarrollo]</div>";
        }
    }
}