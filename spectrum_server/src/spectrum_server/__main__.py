import uvicorn
import logging


def main():
    print("Hello from spectrum_server")
    logging.debug("Debug message from main function")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()  # Log to console
        ],
    )
    logging.debug("Starting Spectrum_Server application")

    uvicorn.run(
        "spectrum_server.server:app", host="0.0.0.0", port=8000, reload=False, log_level="debug"
    )
    main()
