import requests
import urllib3
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, HTML
from io import BytesIO
import io
import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_URL = "https://pad.crc.nd.edu/api/v2"

def get_data_api(request_url, data_type=""):    
    try:
        # fetch_data_from_api
        r = requests.get(url=request_url,verify=False)  # NOTE: Using verify=False due to a SSL issue, I need a valid certificate, then I will remove this parameter.
        r.raise_for_status() # Raise an exception if the status is not 200
        data = r.json()
        df = pd.json_normalize(data)
        return df
    except requests.exceptions.RequestException as e:
        print(e)
        print(f"Error accessing {data_type} data: {r.status_code}")
        return None
    
# Get card issue types
def get_card_issues():
    request_url = f"{API_URL}/cards/issues"
    return get_data_api(request_url, 'card issues')

# Get projects
def get_projects():
    request_url = f"{API_URL}/projects"
    projects = get_data_api(request_url, 'projects')

    # Find columns with all NaN values
    columns_with_all_nan = projects.columns[projects.isnull().all()]

    # Drop columns with all NaN values
    projects = projects.drop(columns=columns_with_all_nan)

    # Check if the column 'sample_names.sample_names' exists in the DataFrame
    if 'sample_names.sample_names' in projects.columns:
        # Rename the column to 'sample_names'
        projects = projects.rename(columns={'sample_names.sample_names': 'sample_names'})

    # Specify the desired column order
    new_column_order = ['id','project_name','annotation','test_name','sample_names','neutral_filler','qpc20','qpc50','qpc80','qpc100','user_name','notes']

    # Reorder the columns in the DataFrame
    projects = projects[new_column_order]

    # Reset the index of the dataframe, dropping the existing index.
    projects = projects.reset_index(drop=True) 

    return projects

# Extended function to get project cards for either a single project ID or multiple project IDs
def get_project_cards(project_ids = None):

    # Get project cards
    def _get_project_cards(project_id):
        request_url = f"{API_URL}/projects/{project_id}/cards"
        return get_data_api(request_url, f"project {project_id} cards")

    # Check if project_ids is None, covert it to a list of all available project
    if project_ids is None:
        project_ids = get_projects().id.tolist()
        
     # Check if project_ids is a single integer, convert it to a list if so
    elif isinstance(project_ids, int):
        project_ids = [project_ids]
    # error    
    elif not isinstance(project_ids, list):
        raise ValueError("project_ids must be a single integer, a list of integers, or None")
    
    all_cards = []  # List to hold dataframes from multiple projects

    for project_id in project_ids:
        # Get cards for each project
        project_cards = _get_project_cards(project_id)
        
        if project_cards is not None:
            all_cards.append(project_cards)
    
    # Concatenate all dataframes into one, if there is data
    if all_cards:
        combined_df = pd.concat(all_cards, ignore_index=True)
        return combined_df
    else:
        print("No data was retrieved for the provided project IDs.")
        return None
        
def get_card(card_id):
    request_url = f"{API_URL}/cards/{card_id}"
    return get_data_api(request_url, f"card {card_id}")

def get_project(project_id):
    request_url = f"{API_URL}/projects/{project_id}"
    return get_data_api(request_url, f"project {project_id}")


# Function to load image from URL
def load_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))
    return img

# Function to create a widget that shows the image and its related data
def create_image_widget_with_info(image_url, data_df):

    small_im_width = 300
    full_im_width = 800
    background_color_field = "#5c6e62"
    background_color_value = "#f9f9f9"
    image_id = data_df.ID.values[0]

    # Create an HTML widget with JavaScript for image zoom on click
    zoomable_image_html = f"""
    <div id="imageContainer_{image_id}">    
      <img id="zoomableImage_{image_id}" src="{image_url}" alt="Image" style="width:{small_im_width}px; cursor: pointer;" 
          onclick="
              var img = document.getElementById('zoomableImage_{image_id}');
              var overlay = document.getElementById('overlay_{image_id}');
              if (img.style.width == '{small_im_width}px') {{
                  img.style.width = '{full_im_width}px';  // Full size image width
                  overlay.style.display = 'flex';  // Show overlay
                  overlay.style.alignItems = 'flex-start';  // Align the image at the top
                  overlay.appendChild(img);  // Move image to overlay
              }} else {{
                  img.style.width = '{small_im_width}px';  // Small size image width
                  document.getElementById('imageContainer_{image_id}').appendChild(img);  // Move image back to grid
                  overlay.style.display = 'none';  // Hide overlay
              }}
          ">
      </div>
      <div id="overlay_{image_id}" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; display: none; background-color: rgba(255,255,255,0.9); z-index: 1000; align-items: flex-start; justify-content: center; overflow: auto;">
      </div>
    """
    
    # Create HTML widget for the zoomable image
    img_widget = widgets.HTML(zoomable_image_html)

    # ID label with left-aligned text, custom color, and bold font style using HTML
    id_label = widgets.HTML(
        "<br>"
    )
    
    # Arrange the clickable image in a vertical box (this will be the first column)
    image_column = widgets.VBox([img_widget])
    # Create a DataFrame-like table using HTML with field names as row headers
    table_style = f"""
    <style>
        table {{
            font-family: sans-serif;
            font-size: 14px;
            border-collapse: collapse;
            width: 500px;
        }}
        td, th {{
            border: 1px solid #dddddd;
            text-align: left;
            padding: 4px;
        }}

        th {{
            background-color: {background_color_field};
            color: white;
            text-align: left;
            width: 120px;
            padding-left: 20px;
        }}
        td{{
            padding-left: 10px;
        }}
        tr:nth-child(even) {{
            background-color: {background_color_value};
        }}
        tr:hover {{
            background-color: #eeeee0;
        }}
    </style>
    """

    table_html = table_style + "<table>"
    for field in data_df.columns:
        
        table_html += "<tr>"
        table_html += f"<th>{field}</th>"
        # if type is bool add icon 
        if data_df[field].dtype == 'bool':
            val = "Yes" if data_df[field].values[0] else "No"
            table_html += f"<td>{val}</td>"
        else:  
            table_html += f"<td>{data_df[field].values[0]}</td>"
        table_html += "</tr>"
    table_html += "</table>"

    # Create HTML widget for the table
    info_table = widgets.HTML(table_html)

    # Arrange the two columns (image and info) side by side in a horizontal box
    columns = widgets.HBox([image_column, info_table])
    
    # Display the ID label above the columns
    return widgets.VBox([id_label, columns])


def show_card(card_id):
    info = get_card(card_id)
    
    if info is None:
        print(f"Failed to retrieve data for card {card_id}")
        return
    
    # Data validation: check if essential fields exist in the API response
    def safe_get(field, default="N/A"):
        try:
            if field in info.columns:
                return info[field].values[0]
            else:
                return default
        except (IndexError, KeyError):
            return default

    # Example of how to use `safe_get` for extracting fields
    data = {
        "ID": [card_id],
        "Sample ID": [safe_get('sample_id')],
        "Sample Name": [safe_get('sample_name')],
        "Quantity": [safe_get('quantity')],
        "Camera Type": [safe_get('camera_type_1')],
        "Issue": [safe_get('issue.name', safe_get('issue'))],
        "Project Name": [safe_get('project.project_name')],
        "Project Id": [safe_get('project.id')],
        "Notes": [safe_get('notes')],
        "Date of Creation": [safe_get('date_of_creation')],
        "Deleted": [safe_get('deleted', default=False)],  # If missing, default to False
    }
    
    # Convert data to DataFrame
    data_df = pd.DataFrame(data)
    
    # Handle missing image URL gracefully
    try:
        image_url = 'https://pad.crc.nd.edu/' + info['processed_file_location'].values[0]
    except (KeyError, IndexError):
        print(f"No valid image found for card {card_id}")
        image_url = 'https://via.placeholder.com/300'  # Default placeholder image
    
    # Create the widget for the image and its info
    image_widget_box = create_image_widget_with_info(image_url, data_df)

    # Display the widget
    display(image_widget_box)

# Function to generate HTML for zoomable images with data from DataFrame
def generate_zoomable_image_html(image_id, sample_id, image_url):
    
    small_im_width = 300
    full_im_width = 600

    return f"""
    <div id="imageContainer_{image_id}">
        <!-- Information above the image -->
        <div style="position: relative; font-size: 14px; color: #5c6e62; margin-bottom: 5px;">
            <strong>ID:</strong> {image_id} <strong>Sample ID:</strong> {sample_id}
        </div>
        <!-- The zoomable image -->        
        <img id="zoomableImage_{image_id}" src="{image_url}" alt="Image" style="width:{small_im_width}px; cursor: pointer;" 
        onclick="
            var img = document.getElementById('zoomableImage_{image_id}');
            var overlay = document.getElementById('overlay_{image_id}');
            if (img.style.width == '{small_im_width}px') {{
                img.style.width = '{full_im_width}px';  // Full size image width
                overlay.style.display = 'flex';  // Show overlay
                overlay.style.alignItems = 'flex-start';  // Align the image at the top
                overlay.appendChild(img);  // Move image to overlay
            }} else {{
                img.style.width = '{small_im_width}px';  // Small size image width
                document.getElementById('imageContainer_{image_id}').appendChild(img);  // Move image back to grid
                overlay.style.display = 'none';  // Hide overlay
            }}
        ">
    </div>
    <div id="overlay_{image_id}" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; display: none; background-color: rgba(255,255,255,0.9); z-index: 1000; align-items: flex-start; justify-content: center; overflow: auto;">
    </div>
    """

# Function to create tabs based on the grouping column and number of images per row
def create_tabs(df, group_column, images_per_row=5):
    # Group the DataFrame by the chosen column
    grouped_data = df.groupby(group_column)

    # Create a list of widgets for each tab (text content + zoomable images)
    items = []
    for group_value, group in grouped_data:
        # Text content at the top for each tab (based on group_column), followed by a horizontal line <hr>
        text_content = widgets.HTML(f"""
        <div style="font-size: 18px; color: #5c6e62;">
            <strong>{group_column.capitalize()}:</strong> {group_value} (#Cards: {len(group)})
        </div>
        <hr style="border: 1px solid #ccc; margin-top: 10px;">
        """)

        # Create a grid of zoomable images for each tab based on the data in the group
        img_widgets = [widgets.HTML(generate_zoomable_image_html(row['id'], row['sample_id'], row['url'])) for _, row in group.iterrows()]
        
        # Create a grid box to hold the images for each group, with a configurable number of images per row
        grid = widgets.GridBox(
            children=img_widgets,
            layout=widgets.Layout(
                grid_template_columns=f'repeat({images_per_row}, 300px)',  # Use the parameter for images per row
                grid_gap='10px'
            )
        )
        
        # Combine text content and grid into a vertical box (VBox)
        combined_content = widgets.VBox([text_content, grid])
        
        # Add the combined content to the list of tab items
        items.append(combined_content)

    # Create the tab widget
    tab = widgets.Tab(children=items)

    # Set tab titles based on the group value and number of Cards
    for i, (group_value, group) in enumerate(grouped_data):
        tab.set_title(i, f"{group_value} ({len(group)})")  # Show group_value and sample count

    # Create an Output widget with fixed height to contain the tab content
    output = widgets.Output(layout=widgets.Layout(height='1000px', overflow_y='auto'))

    # Display the tab widget inside the Output widget with a title
    with output:
        # Adding an HTML title above the tab
        display(widgets.HTML(f"<h2 style='text-align: center;'>Grouped by {group_column.capitalize()}</h2>"))
        display(tab)

    # Display the Output widget with the tabs inside
    display(output)

def show_grouped_cards(df, group_column, images_per_row=5):
    # Add url to dataframe
    df['url'] = df['processed_file_location'].apply(lambda x: f"https://pad.crc.nd.edu/{x}")
    create_tabs(df, group_column, images_per_row)

def create_thumbnail(url, size=(100, 100)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.thumbnail(size)
    return img
    #create_thumbnail('https://pad.crc.nd.edu//var/www/html/images/padimages/processed/40000/42275_processed.png', size=(100, 100))    

def standardize_names(name):
    return name.lower().replace(' ', '-')

# Extended function to get project cards for either a single project ID or multiple project IDs
def get_card_samples(sample_id):

    # Get project cards
    def _get_project_cards(project_id):
        request_url = f"{API_URL}/projects/{project_id}/cards"
        return get_data_api(request_url, f"project {project_id} cards")

    # Check a list of all available project
    project_ids = get_projects().id.tolist()   

    for project_id in project_ids:
        # Get cards for each project
        project_cards = _get_project_cards(project_id)
        
        if project_cards is not None:
          # Filter the cards based on the sample_id
          if 'sample_id' in project_cards.columns:
            card_samples = project_cards[project_cards.sample_id == sample_id]
            
          # Assuming a sample_id can't be in multuples projects
          if not card_samples.empty:
            project_name = get_project(project_id).project_name.values[0]
            # Displaying the message with custom font and color
            display(HTML(f"""
            <div style="font-family: 'Courier New', monospace; color: #1d81df;">
                &#128077; One or more cards with the <strong>Sample ID {sample_id}</strong> were found in the <strong>Project {project_name} (ID={project_id})</strong>                
            </div></br>
            """))            
            return card_samples
    
    # Displaying the message with custom font and dark red color
    display(HTML(f"""
    <div style="font-family: 'Courier New', monospace; color: darkred;">
        &#128308; No data was retrieved for the provided <strong>Sample ID {sample_id}</strong>.
    </div>
    """))
    return None


def show_cards_from_df(cards_df):
    """
    Displays widgets for multiple cards based on the information in the DataFrame.
    Assumes that all necessary fields are present in the DataFrame.
    """
    card_widgets = []
    
    # Iterate through each row in the DataFrame
    for index, row in cards_df.iterrows():
        # Extract the necessary fields from the DataFrame row
        id = row['id']
        sample_id = row.get('sample_id', "N/A")
        sample_name = row.get('sample_name', "N/A")
        quantity = row.get('quantity', "N/A")
        camera_type = row.get('camera_type_1', "N/A")
        issue = row.get('issue', "N/A")
        project_name = row.get('project.project_name', "N/A")
        project_id = row.get('project.id', "N/A")
        notes = row.get('notes', "N/A")
        date_of_creation = row.get('date_of_creation', "N/A")
        deleted = row.get('deleted', False)  # Default to False if not present
        processed_file_location = row.get('processed_file_location', None)

        # Construct the data dictionary for the card
        data = {
            "ID": [id],
            "Sample ID": [sample_id],
            "Sample Name": [sample_name],
            "Quantity": [quantity],
            "Camera Type": [camera_type],
            "Issue": [issue],
            "Project Name": [project_name],
            "Project Id": [project_id],
            "Notes": [notes],
            "Date of Creation": [date_of_creation],
            "Deleted": [deleted],
        }
        data_df = pd.DataFrame(data)
        
        # Generate the image URL, handling the case where it might be missing
        if processed_file_location:
            image_url = f'https://pad.crc.nd.edu/{processed_file_location}'
        else:
            image_url = 'https://via.placeholder.com/300'  # Use placeholder if no image URL

        # Create the widget for this card and add it to the list
        card_widget = create_image_widget_with_info(image_url, data_df)
        card_widgets.append(card_widget)

    # Create a layout to display the cards in a grid-like format
    # Display the widgets in rows of two or three cards per row
    max_cards_per_row = 2  # Adjust how many cards per row
    card_rows = [widgets.HBox(card_widgets[i:i + max_cards_per_row]) for i in range(0, len(card_widgets), max_cards_per_row)]

    # Display the rows of widgets vertically
    display(widgets.VBox(card_rows))

def show_cards(card_ids):
    """
    Displays widgets for multiple cards based on the list of card IDs.
    """
    card_widgets = []
    
    # Iterate through each card in the DataFrame
    for card_id in card_ids:
        # Fetch card data
        info = get_card(card_id)
        
        # Handle the case where the API fails to return the card data
        if info is None:
            # print(f"Failed to retrieve data for card {card_id}")

            # Displaying the message with custom font and dark red color
            display(HTML(f"""
            <div style="font-family: 'Courier New', monospace; color: darkred;">
                &#128308; No data was retrieved for the provided card id lis {card_ids}</strong>.
            </div>
            """))            
            continue
        
        # Safely extract the required fields using the helper function `safe_get`
        def safe_get(field, default="N/A"):
            try:
                if field in info.columns:
                    return info[field].values[0]
                else:
                    return default
            except (IndexError, KeyError):
                return default

        # Prepare the data for the card
        data = {
            "ID": [card_id],
            "Sample ID": [safe_get('sample_id')],
            "Sample Name": [safe_get('sample_name')],
            "Quantity": [safe_get('quantity')],
            "Camera Type": [safe_get('camera_type_1')],
            "Issue": [safe_get('issue.name', safe_get('issue'))],
            "Project Name": [safe_get('project.project_name')],
            "Project Id": [safe_get('project.id')],
            "Notes": [safe_get('notes')],
            "Date of Creation": [safe_get('date_of_creation')],
            "Deleted": [safe_get('deleted', default=False)],
        }

        # Convert to DataFrame for display
        data_df = pd.DataFrame(data)

        # Handle missing image URL safely
        try:
            image_url = 'https://pad.crc.nd.edu/' + info['processed_file_location'].values[0]
        except (KeyError, IndexError):
            print(f"No valid image found for card {card_id}")            
            image_url = 'https://via.placeholder.com/300'  # Placeholder if no image

        # Create the widget for the current card and append it to the list
        card_widget = create_image_widget_with_info(image_url, data_df)
        card_widgets.append(card_widget)

    # Create a layout to display the cards in a grid-like format
    # Display the widgets in rows of two or three cards per row
    max_cards_per_row = 3  # Set how many cards you want per row
    card_rows = [widgets.HBox(card_widgets[i:i + max_cards_per_row]) for i in range(0, len(card_widgets), max_cards_per_row)]

    # Display the rows vertically
    display(widgets.VBox(card_rows))


def get_neural_networks():
    request_url = f"{API_URL}/neural-networks"
    return get_data_api(request_url, 'card issues')

def get_neural_network(nn_id):
    request_url = f"{API_URL}/neural-networks/{nn_id}"
    return get_data_api(request_url, f"neural_network {nn_id}")
    
def read_img(image_url):
  # Get the image data from the URL
  response = requests.get(image_url)
  response.raise_for_status()  # Ensure the request was successful

  # Open the image using PIL directly from the HTTP response
  img = Image.open(BytesIO(response.content))
  return img
