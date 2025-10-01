from pydantic import BaseModel, Field
import datetime

    
class mongoOutput(BaseModel):
    mongoResults: str = Field(...)    
    
class tuneFrequencyOutput(BaseModel):
    sucess: bool = Field(...)

class timeDict(BaseModel):
    nanoseconds: int = Field(...)
    utc_datetime: datetime.datetime = Field(...)

class timeWaitOutput(BaseModel):
    success: bool = Field(...)

class tuneFrequencyOutput(BaseModel):
    success: bool = Field(...)
