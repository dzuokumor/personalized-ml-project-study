const cloudname = import.meta.env.VITE_CLOUDINARY_CLOUD_NAME
const uploadpreset = import.meta.env.VITE_CLOUDINARY_UPLOAD_PRESET

export const uploadimage = async (file) => {
  const formdata = new FormData()
  formdata.append('file', file)
  formdata.append('upload_preset', uploadpreset)
  formdata.append('folder', 'neuron-avatars')

  try {
    const response = await fetch(
      `https://api.cloudinary.com/v1_1/${cloudname}/image/upload`,
      {
        method: 'POST',
        body: formdata
      }
    )

    const data = await response.json()

    if (data.error) {
      throw new Error(data.error.message)
    }

    return {
      url: data.secure_url,
      publicid: data.public_id
    }
  } catch (error) {
    console.error('Upload error:', error)
    throw error
  }
}

export const getoptimizedurl = (url, options = {}) => {
  if (!url || !url.includes('cloudinary')) return url

  const { width = 200, height = 200, crop = 'fill', gravity = 'face' } = options

  const parts = url.split('/upload/')
  if (parts.length !== 2) return url

  return `${parts[0]}/upload/w_${width},h_${height},c_${crop},g_${gravity},q_auto,f_auto/${parts[1]}`
}
