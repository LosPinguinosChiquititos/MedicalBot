CREATE DATABASE ChatbotMedico;
GO

USE ChatbotMedico;
GO

CREATE TABLE Consultas (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    DNI CHAR(8) NOT NULL,
    Nombre NVARCHAR(100),
    Apellido NVARCHAR(100),
    Edad INT,
    Sintomas NVARCHAR(MAX),
    Alergias NVARCHAR(MAX),
    ActividadesPrevias NVARCHAR(MAX),
    EnfermedadesPosibles NVARCHAR(MAX),
    Recomendaciones NVARCHAR(MAX)
);
