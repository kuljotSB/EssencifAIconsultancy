# Essencif.AI Document Intelligence Solution

## Flowchart for PDF Processing Logic

```mermaid
flowchart TD
    A["Start processing PDF page"] --> B["Get invoice data using Azure Document Intelligence"]
    B --> C["Look for important fields:\nInvoice ID, Vendor Name, Total Amount"]
    C --> D{"Does this page have Invoice ID?"}
    
    D -- Yes --> E["Remember this Invoice ID"]
    E --> F{"Different from previous?"}
    F -- Yes --> G["New Invoice Detected\n→ Return 'InvoiceIdChanged'"]
    F -- No --> H["Same invoice continues\n→ Keep current ID"]
    
    D -- No --> I{"Does page have Vendor Name?"}
    I -- Yes --> J["Remember Vendor Name"]
    J --> K{"Different from previous?"}
    K -- Yes --> L["New Vendor Detected\n→ Return 'VendorNameChanged'"]
    K -- No --> M["Same vendor continues"]
    
    I -- No --> N{"Has Total Amount (confident)?"}
    N -- Yes --> O["Likely end of invoice\n→ Return 'InvoiceTotal'"]
    
    N -- No --> P{"Has Customer Name?"}
    P -- Yes --> Q["Likely start of invoice\n→ Return 'InvoiceId'"]
    P -- No --> R["No key fields found\n→ Middle page\n→ Return 'child page'"]
    
    G --> Z["Done"]
    L --> Z
    O --> Z
    Q --> Z
    R --> Z
    M --> Z
    H --> Z
